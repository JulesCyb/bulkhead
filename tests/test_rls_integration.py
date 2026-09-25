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
    """The control-plane schema (#12): `app` gets no INSERT/UPDATE/DELETE anywhere in
    `control`, only SELECT on the exposed read-only view."""
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
    assert set(table_grants) == {("tenants_view", "SELECT")}


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


# --- Gateway credential alias (#52): the control-plane fact + the resolver end to end ------


async def _set_gateway_alias(url: str, tenant_id: uuid.UUID, alias: str) -> None:
    """As the owner role -- the only role that may ever write to `control.tenants` (#12).

    FORCE ROW LEVEL SECURITY (0001/0002) applies to `app_owner` too, so writing still needs
    `app.tenant_id` set to the target tenant, exactly as a real provisioning step would.
    """
    engine = create_async_engine(url)
    async with engine.connect() as conn:
        await conn.execute(
            text("SELECT set_config('app.tenant_id', :t, false)"), {"t": str(tenant_id)}
        )
        await conn.execute(
            text("INSERT INTO control.tenants (tenant_id) VALUES (:tid)"),
            {"tid": tenant_id},
        )
        await conn.execute(
            text(
                "UPDATE control.tenants SET gateway_credential_alias = :alias "
                "WHERE tenant_id = :tid"
            ),
            {"tid": tenant_id, "alias": alias},
        )
        await conn.commit()
    await engine.dispose()


async def test_control_tenants_view_exposes_the_gateway_credential_alias(
    app_settings, database_urls
):
    """The alias the owner role recorded is visible to `app` through the read-only view,
    scoped to the caller's own tenant like every other row in `control.tenants` (#52)."""
    from app.context import RequestContext
    from app.db.session import tenant_session
    from app.repositories.control import ControlRepository

    tenant_a, _ = await _seed(database_urls["superuser"])
    await _set_gateway_alias(database_urls["migrations"], tenant_a, "acme-gateway-key")

    ctx_a = RequestContext(tenant_id=tenant_a, identity_id=uuid.uuid4())
    async with tenant_session(ctx_a) as session:
        alias = await ControlRepository().get_gateway_credential_alias(session, ctx_a)
    assert alias == "acme-gateway-key"


async def test_control_tenants_alias_is_none_when_not_yet_recorded(app_settings, database_urls):
    from app.context import RequestContext
    from app.db.session import tenant_session
    from app.repositories.control import ControlRepository

    tenant_a, _ = await _seed(database_urls["superuser"])
    ctx_a = RequestContext(tenant_id=tenant_a, identity_id=uuid.uuid4())
    async with tenant_session(ctx_a) as session:
        alias = await ControlRepository().get_gateway_credential_alias(session, ctx_a)
    assert alias is None


async def test_app_cannot_write_the_gateway_credential_alias(app_settings, database_urls):
    """`app` has no UPDATE (or any DML) anywhere in `control` -- only the owner role can ever
    set an alias (#12, #52); a tenant's own request cannot forge itself a credential."""
    from sqlalchemy.exc import DBAPIError, ProgrammingError

    from app.context import RequestContext
    from app.db.session import tenant_session

    tenant_a, _ = await _seed(database_urls["superuser"])
    await _set_gateway_alias(database_urls["migrations"], tenant_a, "acme-gateway-key")
    ctx_a = RequestContext(tenant_id=tenant_a, identity_id=uuid.uuid4())
    with pytest.raises((DBAPIError, ProgrammingError)):
        async with tenant_session(ctx_a) as session:
            await session.execute(
                text(
                    "UPDATE control.tenants SET gateway_credential_alias = 'forged' "
                    "WHERE tenant_id = :tid"
                ),
                {"tid": tenant_a},
            )


async def test_two_tenants_with_different_aliases_resolve_different_credentials(
    app_settings, database_urls, tmp_path
):
    """End to end (#52): given a tenant id, the resolver reads the control plane's alias, then
    that alias's secret file fresh -- two tenants with different aliases get two different
    credential contents, nothing shared or cached between them."""
    from app.config import Settings
    from app.context import RequestContext
    from app.db.session import tenant_session
    from app.gateway_credentials import resolve_gateway_credential

    tenant_a, tenant_b = await _seed(database_urls["superuser"])
    await _set_gateway_alias(database_urls["migrations"], tenant_a, "acme-gateway-key")
    await _set_gateway_alias(database_urls["migrations"], tenant_b, "globex-gateway-key")
    (tmp_path / "acme-gateway-key").write_text("sk-acme-secret")
    (tmp_path / "globex-gateway-key").write_text("sk-globex-secret")
    settings = Settings(
        database_url="postgresql+asyncpg://app:app@localhost:5432/app",
        gateway_credentials_dir=str(tmp_path),
    )

    ctx_a = RequestContext(tenant_id=tenant_a, identity_id=uuid.uuid4())
    async with tenant_session(ctx_a) as session:
        credential_a = await resolve_gateway_credential(session, ctx_a, settings=settings)

    ctx_b = RequestContext(tenant_id=tenant_b, identity_id=uuid.uuid4())
    async with tenant_session(ctx_b) as session:
        credential_b = await resolve_gateway_credential(session, ctx_b, settings=settings)

    assert credential_a.get_secret_value() == "sk-acme-secret"
    assert credential_b.get_secret_value() == "sk-globex-secret"


async def test_resolver_raises_typed_error_for_a_tenant_with_no_alias_recorded(
    app_settings, database_urls
):
    from app.context import RequestContext
    from app.db.session import tenant_session
    from app.gateway_credentials import GatewayCredentialUnavailable, resolve_gateway_credential

    tenant_a, _ = await _seed(database_urls["superuser"])
    ctx_a = RequestContext(tenant_id=tenant_a, identity_id=uuid.uuid4())
    with pytest.raises(GatewayCredentialUnavailable):
        async with tenant_session(ctx_a) as session:
            await resolve_gateway_credential(session, ctx_a)


async def test_resolver_raises_typed_error_when_the_secret_file_is_missing(
    app_settings, database_urls, tmp_path
):
    from app.config import Settings
    from app.context import RequestContext
    from app.db.session import tenant_session
    from app.gateway_credentials import GatewayCredentialUnavailable, resolve_gateway_credential

    tenant_a, _ = await _seed(database_urls["superuser"])
    await _set_gateway_alias(database_urls["migrations"], tenant_a, "acme-gateway-key")
    settings = Settings(
        database_url="postgresql+asyncpg://app:app@localhost:5432/app",
        gateway_credentials_dir=str(tmp_path),
    )

    ctx_a = RequestContext(tenant_id=tenant_a, identity_id=uuid.uuid4())
    with pytest.raises(GatewayCredentialUnavailable):
        async with tenant_session(ctx_a) as session:
            await resolve_gateway_credential(session, ctx_a, settings=settings)
