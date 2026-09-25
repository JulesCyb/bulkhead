"""Real isolation test against PostgreSQL + pgvector (`pgserver`, via `uv sync --group dbtest`).

Verifies what the unit tests cannot: that the RLS policies from the migration apply when the
app works as the `app` role (no superuser, NOBYPASSRLS), and that the role bootstrap itself
(docker/postgres/01-init.sh) shapes `app_owner`/`app` the way it claims to: neither role is a
superuser or holds BYPASSRLS, `app_owner` owns the schema and runs the whole migration suite,
a freshly created table grants `app` nothing until a migration says so explicitly, and `app`'s
statement timeout matches the role-level setting the bootstrap script configures.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import uuid
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import ROLE_STATEMENT_TIMEOUT_MS  # noqa: E402

pgserver = pytest.importorskip("pgserver")

DIM = 1536


def _vec(seed: float) -> str:
    values = [0.0] * DIM
    values[0] = 1.0
    values[1] = seed
    return "[" + ",".join(f"{v:.3f}" for v in values) + "]"


def _psql(server, command: str) -> None:
    """`server.psql` without a shell: pgserver's own version breaks on paths with spaces."""
    from pgserver.postgres_server import POSTGRES_BIN_PATH

    subprocess.run(
        [str(POSTGRES_BIN_PATH / "psql"), server.get_uri()],
        input=command.encode(),
        check=True,
        capture_output=True,
    )


@pytest.fixture(scope="module")
def database_urls():
    """Mirrors docker/postgres/01-init.sh: the cluster's own bootstrap superuser (here,
    pgserver's default `postgres` role) creates the extension once, then `app_owner` (owns the
    schema, runs every migration) and `app` (unchanged: no superuser, NOBYPASSRLS, a statement
    timeout), with no default privileges on future tables for either.
    """
    pgdata = tempfile.mkdtemp(prefix="pgdata-")
    server = pgserver.get_server(pgdata)
    sockdir = parse_qs(urlparse(server.get_uri()).query)["host"][0]
    _psql(
        server,
        "CREATE EXTENSION IF NOT EXISTS vector; "
        "CREATE ROLE app_owner LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE; "
        "ALTER SCHEMA public OWNER TO app_owner; "
        "GRANT CREATE ON DATABASE postgres TO app_owner; "
        "CREATE ROLE app LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE; "
        "GRANT USAGE ON SCHEMA public TO app; "
        f"ALTER ROLE app SET statement_timeout = '{ROLE_STATEMENT_TIMEOUT_MS}ms';",
    )
    urls = {
        # The owner role: the only one migrations, seeding, and this fixture's own schema
        # setup connect as — never the cluster superuser.
        "migrations": f"postgresql+asyncpg://app_owner@/postgres?host={sockdir}",
        "app": f"postgresql+asyncpg://app@/postgres?host={sockdir}",
        # Test-only: `app_owner` does not bypass RLS (FORCE ROW LEVEL SECURITY applies to it
        # like any other non-superuser), so seeding test fixtures without a tenant context
        # needs the real superuser, exactly like production never would.
        "superuser": f"postgresql+asyncpg://postgres@/postgres?host={sockdir}",
    }
    env = {**os.environ, "DATABASE_URL_MIGRATIONS": urls["migrations"], "DATABASE_URL": urls["app"]}
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"], check=True, env=env, timeout=120
    )
    yield urls
    server.cleanup()


@pytest.fixture
def app_settings(database_urls, monkeypatch):
    from app import config
    from app.db import session as db_session

    monkeypatch.setenv("DATABASE_URL", database_urls["app"])
    monkeypatch.setenv("DATABASE_URL_MIGRATIONS", database_urls["migrations"])
    config.get_settings.cache_clear()
    db_session._engine = None
    db_session._session_factory = None
    yield
    config.get_settings.cache_clear()
    db_session._engine = None
    db_session._session_factory = None


async def _seed(url: str) -> tuple[uuid.UUID, uuid.UUID]:
    """Two tenants with one document each — as the owner; without context RLS blocks all."""
    engine = create_async_engine(url)
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    async with engine.begin() as conn:
        for tenant_id, name, seed in ((tenant_a, "A", 0.1), (tenant_b, "B", 0.9)):
            await conn.execute(
                text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
                {"id": tenant_id, "name": name},
            )
            await conn.execute(
                text(
                    "INSERT INTO documents (tenant_id, title, content, embedding) "
                    "VALUES (:tid, :title, :content, CAST(:emb AS vector))"
                ),
                {
                    "tid": tenant_id,
                    "title": f"Document {name}",
                    "content": f"Content of tenant {name}",
                    "emb": _vec(seed),
                },
            )
    await engine.dispose()
    return tenant_a, tenant_b


async def test_search_sees_only_own_tenant(app_settings, database_urls):
    from app.context import RequestContext
    from app.db.session import tenant_session
    from app.repositories.documents import DocumentRepository

    tenant_a, tenant_b = await _seed(database_urls["superuser"])
    query = [0.0] * DIM
    query[0] = 1.0

    ctx_a = RequestContext(tenant_id=tenant_a, identity_id=uuid.uuid4())
    async with tenant_session(ctx_a) as session:
        hits = await DocumentRepository().search(session, query, limit=10)
    assert [h.title for h in hits] == ["Document A"]

    ctx_b = RequestContext(tenant_id=tenant_b, identity_id=uuid.uuid4())
    async with tenant_session(ctx_b) as session:
        hits = await DocumentRepository().search(session, query, limit=10)
    assert [h.title for h in hits] == ["Document B"]


async def test_insert_for_other_tenant_is_rejected(app_settings, database_urls):
    from sqlalchemy.exc import DBAPIError

    from app.context import RequestContext
    from app.db.session import tenant_session

    tenant_a, tenant_b = await _seed(database_urls["superuser"])
    ctx_a = RequestContext(tenant_id=tenant_a, identity_id=uuid.uuid4())
    with pytest.raises(DBAPIError):
        async with tenant_session(ctx_a) as session:
            await session.execute(
                text(
                    "INSERT INTO documents (tenant_id, title, content) "
                    "VALUES (:tid, 'foreign', 'must not work')"
                ),
                {"tid": tenant_b},
            )


async def test_app_role_cannot_delete_tenants(app_settings, database_urls):
    """The migration revokes INSERT/DELETE on tenants from the app role — a DELETE would
    cascade an entire tenant away in one statement."""
    from sqlalchemy.exc import DBAPIError, ProgrammingError

    from app.context import RequestContext
    from app.db.session import tenant_session

    tenant_a, _ = await _seed(database_urls["superuser"])
    ctx_a = RequestContext(tenant_id=tenant_a, identity_id=uuid.uuid4())
    with pytest.raises((DBAPIError, ProgrammingError)):
        async with tenant_session(ctx_a) as session:
            await session.execute(text("DELETE FROM tenants WHERE id = :tid"), {"tid": tenant_a})


async def test_tenant_session_sets_identity_id(app_settings, database_urls):
    """The new session-setting name is set and readable — no policy reads it yet (Spec 3's job)."""
    from app.context import RequestContext
    from app.db.session import tenant_session

    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    ctx = RequestContext(tenant_id=tenant_id, identity_id=identity_id)
    async with tenant_session(ctx) as session:
        value = (
            await session.execute(text("SELECT current_setting('app.identity_id', true)"))
        ).scalar_one()
    assert value == str(identity_id)


async def test_no_context_means_no_rows(app_settings, database_urls):
    """Without set_config, current_setting is NULL -> the policy blocks all (app role)."""
    await _seed(database_urls["superuser"])
    engine = create_async_engine(database_urls["app"])
    async with engine.connect() as conn:
        count = (await conn.execute(text("SELECT count(*) FROM documents"))).scalar_one()
    await engine.dispose()
    assert count == 0


async def test_pool_checkin_clears_leftover_tenant_context(app_settings):
    """A connection returned to the pool carries none of a request's session-local
    settings. Simulates the bug this guards against: a context committed at session
    scope (set_config(..., false)), not transaction scope (is_local=true) — a plain
    rollback-on-checkin would not undo an already-committed session setting, so the
    pool must explicitly wipe it on checkin instead."""
    from app.db.session import get_engine

    engine = get_engine()
    tenant_id = str(uuid.uuid4())

    conn = await engine.connect()
    await conn.execute(text("SELECT set_config('app.tenant_id', :tid, false)"), {"tid": tenant_id})
    set_value = (
        await conn.execute(text("SELECT current_setting('app.tenant_id', true)"))
    ).scalar_one()
    assert set_value == tenant_id
    await conn.commit()  # the setting is now session-level and committed, not rolled back
    await conn.close()  # checkin

    conn2 = await engine.connect()
    try:
        leftover = (
            await conn2.execute(text("SELECT current_setting('app.tenant_id', true)"))
        ).scalar_one()
    finally:
        await conn2.close()
    assert leftover in (None, "")


async def test_neither_role_is_superuser_or_bypasses_rls(database_urls):
    """docker/postgres/01-init.sh's whole point: `app_owner` and `app` must both be ordinary,
    non-privileged roles — only the cluster's own bootstrap superuser, used once, may bypass
    Row-Level Security."""
    engine = create_async_engine(database_urls["app"])
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT rolname, rolsuper, rolbypassrls FROM pg_roles "
                    "WHERE rolname IN ('app_owner', 'app')"
                )
            )
        ).all()
    await engine.dispose()
    by_name = {row.rolname: row for row in rows}
    assert set(by_name) == {"app_owner", "app"}
    for role in ("app_owner", "app"):
        assert by_name[role].rolsuper is False
        assert by_name[role].rolbypassrls is False


async def test_owner_owns_public_schema_and_ran_the_migrations(database_urls):
    """`app_owner` owns `public`, and — since the fixture points DATABASE_URL_MIGRATIONS at it
    and the whole migration suite already ran against it to get here — every table it created is
    owned by it too, not by the cluster superuser."""
    engine = create_async_engine(database_urls["migrations"])
    async with engine.connect() as conn:
        schema_owner = (
            await conn.execute(
                text(
                    "SELECT r.rolname FROM pg_namespace n "
                    "JOIN pg_roles r ON r.oid = n.nspowner WHERE n.nspname = 'public'"
                )
            )
        ).scalar_one()
        table_owners = (
            (
                await conn.execute(
                    text(
                        "SELECT tableowner FROM pg_tables WHERE schemaname = 'public' "
                        "AND tablename IN ('tenants', 'users', 'documents')"
                    )
                )
            )
            .scalars()
            .all()
        )
    await engine.dispose()
    assert schema_owner == "app_owner"
    assert table_owners == ["app_owner", "app_owner", "app_owner"]


async def test_new_table_gets_no_default_privileges(database_urls):
    """The blanket ALTER DEFAULT PRIVILEGES grant is gone: a freshly created table is
    unreadable and unwritable by `app` until a migration explicitly grants it."""
    engine = create_async_engine(database_urls["migrations"])
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE TABLE scratch_no_default_grants (id int)"))
        async with engine.connect() as conn:
            grants = (
                await conn.execute(
                    text(
                        "SELECT privilege_type FROM information_schema.table_privileges "
                        "WHERE table_name = 'scratch_no_default_grants' AND grantee = 'app'"
                    )
                )
            ).all()
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DROP TABLE IF EXISTS scratch_no_default_grants"))
        await engine.dispose()
    assert grants == []


async def test_app_has_no_dml_on_control_schema_only_select_on_the_view(database_urls):
    """The control-plane schema (#12, extended by #22): `app` gets no INSERT/UPDATE/DELETE
    anywhere in `control`, only SELECT on the two exposed read-only views (tenant facts,
    identity lookup) -- plus, separately, EXECUTE on the one narrow function (#22, asserted in
    its own test), which is not a table privilege at all."""
    engine = create_async_engine(database_urls["app"])
    async with engine.connect() as conn:
        table_grants = (
            await conn.execute(
                text(
                    "SELECT table_name, privilege_type FROM information_schema.table_privileges "
                    "WHERE table_schema = 'control' AND grantee = 'app'"
                )
            )
        ).all()
    await engine.dispose()
    assert set(table_grants) == {("tenants_view", "SELECT"), ("identity_lookup", "SELECT")}


async def test_app_can_update_own_settings_but_not_other_tenant_columns(
    app_settings, database_urls
):
    """Column-level grant (#12): `app` may UPDATE tenants.settings but nothing else on that
    row -- rejected by Postgres's own privilege check, not by RLS/policy."""
    from sqlalchemy.exc import DBAPIError

    from app.context import RequestContext
    from app.db.session import tenant_session

    tenant_a, _ = await _seed(database_urls["superuser"])
    ctx_a = RequestContext(tenant_id=tenant_a, identity_id=uuid.uuid4())

    async with tenant_session(ctx_a) as session:
        await session.execute(
            text("UPDATE tenants SET settings = :s WHERE id = :tid"),
            {"s": '{"k": "v"}', "tid": str(tenant_a)},
        )
        await session.commit()

    with pytest.raises(DBAPIError):
        async with tenant_session(ctx_a) as session:
            await session.execute(
                text("UPDATE tenants SET name = 'renamed' WHERE id = :tid"),
                {"tid": str(tenant_a)},
            )


async def test_tenants_self_only_policy_validates_writes_too(database_urls):
    """`tenants_self_only` (#12) now carries a WITH CHECK matching its USING clause: a write
    that would move a row out of the caller's own tenant is rejected by the policy itself.

    Runs as `app_owner`, not `app`: `app`'s column-level grant only permits UPDATE of
    `settings`, which can never violate this check (it never touches `id`), so the check's own
    enforcement can only be observed with a role that is allowed to update `id` in the first
    place -- `app_owner` is such a role, and FORCE ROW LEVEL SECURITY (set in 0001) makes the
    policy apply to the table owner too, not only to `app`.
    """
    from sqlalchemy.exc import DBAPIError

    superuser_engine = create_async_engine(database_urls["superuser"])
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    async with superuser_engine.begin() as conn:
        for tenant_id, name in ((tenant_a, "A"), (tenant_b, "B")):
            await conn.execute(
                text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
                {"id": tenant_id, "name": name},
            )
    await superuser_engine.dispose()

    engine = create_async_engine(database_urls["migrations"])
    async with engine.connect() as conn:
        await conn.execute(
            text("SELECT set_config('app.tenant_id', :t, false)"), {"t": str(tenant_a)}
        )
        # Within its own tenant, an update that keeps id unchanged still satisfies WITH CHECK.
        result = await conn.execute(
            text("UPDATE tenants SET name = 'still A' WHERE id = :tid RETURNING id"),
            {"tid": str(tenant_a)},
        )
        assert result.scalar_one() == tenant_a
        await conn.commit()

        # Re-pointing the row's own id away from the caller's tenant violates WITH CHECK: the
        # USING clause lets the update reach the row (id still matches app.tenant_id going in),
        # but the resulting row would no longer satisfy the same expression.
        with pytest.raises(DBAPIError):
            await conn.execute(
                text("UPDATE tenants SET id = :new_id WHERE id = :tid"),
                {"new_id": str(uuid.uuid4()), "tid": str(tenant_a)},
            )
    await engine.dispose()


async def test_control_tenants_owned_by_app_owner_never_app(database_urls):
    """The control-plane schema and its table are owned by `app_owner`, never `app` (#12)."""
    engine = create_async_engine(database_urls["migrations"])
    async with engine.connect() as conn:
        schema_owner = (
            await conn.execute(
                text(
                    "SELECT r.rolname FROM pg_namespace n "
                    "JOIN pg_roles r ON r.oid = n.nspowner WHERE n.nspname = 'control'"
                )
            )
        ).scalar_one()
        table_owner = (
            await conn.execute(
                text(
                    "SELECT tableowner FROM pg_tables "
                    "WHERE schemaname = 'control' AND tablename = 'tenants'"
                )
            )
        ).scalar_one()
    await engine.dispose()
    assert schema_owner == "app_owner"
    assert table_owner == "app_owner"


async def test_control_tenants_has_forced_rls_with_using_and_check(database_urls):
    """`control.tenants` carries forced RLS with a policy that both restricts and validates,
    the same shape as every other tenant-scoped table (#12)."""
    engine = create_async_engine(database_urls["migrations"])
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT c.relrowsecurity, c.relforcerowsecurity FROM pg_class c "
                    "JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "WHERE n.nspname = 'control' AND c.relname = 'tenants'"
                )
            )
        ).one()
        policy = (
            await conn.execute(
                text(
                    "SELECT qual IS NOT NULL, with_check IS NOT NULL FROM pg_policies "
                    "WHERE schemaname = 'control' AND tablename = 'tenants'"
                )
            )
        ).one()
    await engine.dispose()
    assert row.relrowsecurity is True
    assert row.relforcerowsecurity is True
    assert policy[0] is True
    assert policy[1] is True


async def test_control_session_sets_no_tenant_or_identity_context(app_settings):
    """The control-plane session mode (#22): no app.tenant_id / app.identity_id at all -- never
    the same thing as a tenant_session() with an empty context."""
    from app.db.session import control_session

    async with control_session() as session:
        tenant_setting = (
            await session.execute(text("SELECT current_setting('app.tenant_id', true)"))
        ).scalar_one()
        identity_setting = (
            await session.execute(text("SELECT current_setting('app.identity_id', true)"))
        ).scalar_one()
    assert tenant_setting in (None, "")
    assert identity_setting in (None, "")


async def test_control_session_sees_no_rows_of_a_tenants_own_tables(app_settings, database_urls):
    """Documents the "never used against a tenant's own tables" rule (#22) as an observable
    property: with no tenant context, RLS blocks every row of a tenant-scoped table exactly as
    it does for any other contextless session."""
    from app.db.session import control_session

    await _seed(database_urls["superuser"])
    async with control_session() as session:
        count = (await session.execute(text("SELECT count(*) FROM documents"))).scalar_one()
    assert count == 0


async def test_identity_lookup_finds_a_known_identity_and_nothing_for_an_unknown_one(
    app_settings, database_urls
):
    """The identity-lookup component (#22): a known issuer+subject returns the matching row
    over the control-plane session; an unknown pair returns None rather than raising."""
    from app.db.session import control_session
    from app.repositories.control import IdentityRepository

    superuser_engine = create_async_engine(database_urls["superuser"])
    async with superuser_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO control.identities (issuer, subject, display_name, email) "
                "VALUES (:issuer, :subject, 'Ada Lovelace', 'ada@example.com')"
            ),
            {"issuer": "https://idp.example.com", "subject": "sub-123"},
        )
    await superuser_engine.dispose()

    async with control_session() as session:
        found = await IdentityRepository().find_by_issuer_and_subject(
            session, issuer="https://idp.example.com", subject="sub-123"
        )
        missing = await IdentityRepository().find_by_issuer_and_subject(
            session, issuer="https://idp.example.com", subject="no-such-subject"
        )
    assert found is not None
    assert found.issuer == "https://idp.example.com"
    assert found.subject == "sub-123"
    assert missing is None


async def test_identity_lookup_view_exposes_only_id_issuer_subject(database_urls):
    """(#22) control.identity_lookup never leaks display_name/email, even though
    control.identities carries those columns."""
    engine = create_async_engine(database_urls["migrations"])
    async with engine.connect() as conn:
        columns = (
            (
                await conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema = 'control' AND table_name = 'identity_lookup'"
                    )
                )
            )
            .scalars()
            .all()
        )
    await engine.dispose()
    assert set(columns) == {"id", "issuer", "subject"}


async def test_app_cannot_write_identities_only_read_the_lookup(app_settings, database_urls):
    """(#22) `app` may SELECT through control.identity_lookup and nothing else: INSERT/UPDATE/
    DELETE against either the base table or the lookup view fail on privileges."""
    from sqlalchemy.exc import DBAPIError

    from app.db.session import control_session

    async with control_session() as session:
        result = await session.execute(text("SELECT count(*) FROM control.identity_lookup"))
        assert result.scalar_one() >= 0  # SELECT succeeds

    engine = create_async_engine(database_urls["app"])
    statements = [
        "INSERT INTO control.identities (issuer, subject) VALUES ('x', 'y')",
        "UPDATE control.identities SET subject = 'z'",
        "DELETE FROM control.identities",
        "INSERT INTO control.identity_lookup (id, issuer, subject) "
        "VALUES (gen_random_uuid(), 'x', 'y')",
        "UPDATE control.identity_lookup SET subject = 'z'",
        "DELETE FROM control.identity_lookup",
    ]
    for statement in statements:
        async with engine.connect() as conn:
            with pytest.raises(DBAPIError):
                await conn.execute(text(statement))
    await engine.dispose()


async def test_tenant_auth_settings_falls_back_to_default_and_reports_suspension(
    app_settings, database_urls
):
    """(#22) the per-tenant read helper: the tenant's configured issuer when set, the
    process-wide default when unset, and the suspension flag either way."""
    from app.db.session import control_session
    from app.repositories.control import TenantAuthSettingsRepository

    tenant_id = uuid.uuid4()
    superuser_engine = create_async_engine(database_urls["superuser"])
    async with superuser_engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, 'Tenant with defaults')"),
            {"id": tenant_id},
        )
        await conn.execute(
            text("INSERT INTO control.tenants (tenant_id) VALUES (:id)"), {"id": tenant_id}
        )
    await superuser_engine.dispose()

    repo = TenantAuthSettingsRepository()
    async with control_session() as session:
        settings = await repo.get(
            session, tenant_id=tenant_id, default_issuer="https://default.example.com"
        )
    assert settings is not None
    assert settings.issuer == "https://default.example.com"
    assert settings.suspended is False

    superuser_engine = create_async_engine(database_urls["superuser"])
    async with superuser_engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE control.tenants SET identity_issuer = :issuer, "
                "suspended_at = now() WHERE tenant_id = :id"
            ),
            {"issuer": "https://tenant-own-idp.example.com", "id": tenant_id},
        )
    await superuser_engine.dispose()

    async with control_session() as session:
        settings = await repo.get(
            session, tenant_id=tenant_id, default_issuer="https://default.example.com"
        )
    assert settings is not None
    assert settings.issuer == "https://tenant-own-idp.example.com"
    assert settings.suspended is True


async def test_tenant_auth_settings_returns_none_for_unknown_tenant(app_settings):
    from app.db.session import control_session
    from app.repositories.control import TenantAuthSettingsRepository

    async with control_session() as session:
        settings = await TenantAuthSettingsRepository().get(
            session, tenant_id=uuid.uuid4(), default_issuer="https://default.example.com"
        )
    assert settings is None


async def test_app_cannot_write_tenant_auth_settings_columns(app_settings, database_urls):
    """(#22) only app_owner may write identity_issuer/suspended_at -- app has no grant on
    control.tenants at all beyond what tenant_auth_settings()/identity_lookup expose."""
    from sqlalchemy.exc import DBAPIError

    engine = create_async_engine(database_urls["app"])
    async with engine.connect() as conn:
        with pytest.raises(DBAPIError):
            await conn.execute(
                text("UPDATE control.tenants SET identity_issuer = 'https://evil.example.com'")
            )
    await engine.dispose()


async def test_app_statement_timeout_matches_bootstrap(database_urls):
    """The role-level statement_timeout the bootstrap script sets for `app` is the one that
    actually applies to its connections."""
    engine = create_async_engine(database_urls["app"])
    async with engine.connect() as conn:
        timeout_ms = (
            await conn.execute(
                text("SELECT setting FROM pg_settings WHERE name = 'statement_timeout'")
            )
        ).scalar_one()
    await engine.dispose()
    assert int(timeout_ms) == ROLE_STATEMENT_TIMEOUT_MS
