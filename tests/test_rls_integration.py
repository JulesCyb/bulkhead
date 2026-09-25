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
    server = pgserver.get_server(pgdata, cleanup_mode="delete")
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
    """Two tenants with one document each — as the owner; without context RLS blocks all.

    Documents' created_by/updated_by (#29) are NOT NULL foreign keys to control.identities,
    defaulted from the session's app.identity_id -- so seeding needs a real identity row and
    that setting in scope before the INSERTs, exactly like a real tenant_session() would supply.
    """
    engine = create_async_engine(url)
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    seed_identity = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO control.identities (id, issuer, subject) VALUES (:id, 'seed', :sub)"),
            {"id": seed_identity, "sub": str(seed_identity)},
        )
        await conn.execute(
            text("SELECT set_config('app.identity_id', :iid, true)"),
            {"iid": str(seed_identity)},
        )
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


async def _create_tenant_and_identity(url: str) -> tuple[uuid.UUID, uuid.UUID]:
    """A bare tenant and a real control.identities row — for the document audit-column tests
    (#29), whose created_by/updated_by are NOT NULL foreign keys to control.identities."""
    engine = create_async_engine(url)
    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, 'Audit')"), {"id": tenant_id}
        )
        await conn.execute(
            text("INSERT INTO control.identities (id, issuer, subject) VALUES (:id, 'seed', :sub)"),
            {"id": identity_id, "sub": str(identity_id)},
        )
    await engine.dispose()
    return tenant_id, identity_id


async def test_document_insert_sets_created_by_from_session_identity(app_settings, database_urls):
    """#29 AC1: inserting a document through the tenant-bound session sets created_by (and
    updated_by, on first write) to the session's identity without application code passing it
    explicitly -- DocumentRepository.add() never mentions the column."""
    from app.context import RequestContext
    from app.db.session import tenant_session
    from app.repositories.documents import DocumentRepository

    tenant_id, identity_id = await _create_tenant_and_identity(database_urls["superuser"])
    ctx = RequestContext(tenant_id=tenant_id, identity_id=identity_id)
    async with tenant_session(ctx) as session:
        doc = await DocumentRepository().add(session, ctx, title="t", content="c", embedding=None)
        doc_id = doc.id

    # Re-read the row directly: the property under test is what the database persisted, not
    # what the ORM's local object happens to reflect.
    engine = create_async_engine(database_urls["superuser"])
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text("SELECT created_by, updated_by FROM documents WHERE id = :id"),
                {"id": doc_id},
            )
        ).one()
    await engine.dispose()
    assert row.created_by == identity_id
    assert row.updated_by == identity_id


async def test_document_update_refreshes_updated_by_and_keeps_created_by(
    app_settings, database_urls
):
    """#29 AC2: updating a document in a second transaction with a different session identity
    changes updated_by/updated_at, while the original creator (created_by) stays unchanged."""
    from app.context import RequestContext
    from app.db.session import tenant_session
    from app.repositories.documents import DocumentRepository

    tenant_id, creator_id = await _create_tenant_and_identity(database_urls["superuser"])
    _, editor_id = await _create_tenant_and_identity(database_urls["superuser"])

    creator_ctx = RequestContext(tenant_id=tenant_id, identity_id=creator_id)
    async with tenant_session(creator_ctx) as session:
        doc = await DocumentRepository().add(
            session, creator_ctx, title="t", content="c", embedding=None
        )
        doc_id = doc.id

    editor_ctx = RequestContext(tenant_id=tenant_id, identity_id=editor_id)
    async with tenant_session(editor_ctx) as session:
        await session.execute(
            text("UPDATE documents SET title = 'updated' WHERE id = :id"), {"id": doc_id}
        )

    engine = create_async_engine(database_urls["superuser"])
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT created_by, updated_by, created_at, updated_at "
                    "FROM documents WHERE id = :id"
                ),
                {"id": doc_id},
            )
        ).one()
    await engine.dispose()
    assert row.created_by == creator_id
    assert row.updated_by == editor_id
    assert row.updated_at > row.created_at


async def test_document_created_by_fk_checked_as_owner_not_app_role(app_settings, database_urls):
    """#29 AC3: the creator FK resolves against control.identities even though the app role's
    only grant on it is the narrow issuer-plus-subject lookup view (ADR-0003) -- app has no
    grant at all on control.identities itself. Proves the FK is checked with the referenced
    table's owner privileges, not the querying role's, rather than assuming it: a real identity
    resolves despite the missing grant, and an unknown identity is still rejected by the FK."""
    from sqlalchemy.exc import DBAPIError

    from app.context import RequestContext
    from app.db.session import tenant_session
    from app.repositories.documents import DocumentRepository

    tenant_id, identity_id = await _create_tenant_and_identity(database_urls["superuser"])

    engine = create_async_engine(database_urls["app"])
    async with engine.connect() as conn:
        grants = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM information_schema.role_table_grants "
                    "WHERE table_schema = 'control' AND table_name = 'identities' "
                    "AND grantee = current_user"
                )
            )
        ).scalar_one()
    await engine.dispose()
    assert grants == 0

    ctx = RequestContext(tenant_id=tenant_id, identity_id=identity_id)
    async with tenant_session(ctx) as session:
        doc = await DocumentRepository().add(session, ctx, title="t", content="c", embedding=None)
        doc_id = doc.id

    engine = create_async_engine(database_urls["superuser"])
    async with engine.connect() as conn:
        created_by = (
            await conn.execute(
                text("SELECT created_by FROM documents WHERE id = :id"), {"id": doc_id}
            )
        ).scalar_one()
    await engine.dispose()
    assert created_by == identity_id

    unknown_ctx = RequestContext(tenant_id=tenant_id, identity_id=uuid.uuid4())
    with pytest.raises(DBAPIError):
        async with tenant_session(unknown_ctx) as session:
            await DocumentRepository().add(
                session, unknown_ctx, title="t2", content="c2", embedding=None
            )


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
                        "AND tablename IN ('tenants', 'memberships', 'documents')"
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
    """The control-plane schema (#12, extended by #22, #77): `app` gets no INSERT/UPDATE/DELETE
    anywhere in `control`, only SELECT on the three exposed read-only views (tenant facts,
    identity lookup, database-alias enumeration for the runtime guard) -- plus, separately,
    EXECUTE on the one narrow function (#22, asserted in its own test), which is not a table
    privilege at all."""
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
    assert set(table_grants) == {
        ("tenants_view", "SELECT"),
        ("identity_lookup", "SELECT"),
    }


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
    the same shape as every other tenant-scoped table (#12). Selected by name, not "the only
    policy on this table": migration 0016 (#76) adds a second, purely additive SELECT-only
    policy (a narrow migration-runner escape hatch) alongside this one -- it carries no
    with_check clause of its own and does not change this policy's shape."""
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
                    "WHERE schemaname = 'control' AND tablename = 'tenants' "
                    "AND policyname = 'control_tenants_tenant_isolation'"
                )
            )
        ).one()
    await engine.dispose()
    assert row.relrowsecurity is True
    assert row.relforcerowsecurity is True
    assert policy[0] is True
    assert policy[1] is True


async def _public_schema_tenant_isolation_violations(conn, exceptions: frozenset[str]) -> dict:
    """Walks live catalog state (not a fixed list of table names) and reports, per table
    outside `exceptions`, which of the three mandatory parts (forced RLS, a policy that both
    restricts and validates, tied to `app.tenant_id`) is missing. An empty result means every
    table in `public` outside the exception list is fully protected.
    """
    tables = (
        await conn.execute(
            text(
                "SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = 'public' AND c.relkind = 'r'"
            )
        )
    ).all()

    violations: dict[str, list[str]] = {}
    for row in tables:
        if row.relname in exceptions:
            continue
        reasons = []
        if not row.relrowsecurity:
            reasons.append("row security not enabled")
        if not row.relforcerowsecurity:
            reasons.append("row security not forced")

        policies = (
            await conn.execute(
                text(
                    "SELECT qual, with_check FROM pg_policies "
                    "WHERE schemaname = 'public' AND tablename = :t"
                ),
                {"t": row.relname},
            )
        ).all()
        # A policy both restricts (USING -> qual) and validates (WITH CHECK -> with_check)
        # against the tenant setting; the column it compares (tenant_id on most tables, id on
        # `tenants` itself) varies, but every such policy names the same setting.
        has_tenant_policy = any(
            p.qual
            and "app.tenant_id" in p.qual
            and p.with_check
            and "app.tenant_id" in p.with_check
            for p in policies
        )
        if not has_tenant_policy:
            reasons.append(
                "no policy with both a restrict (USING) and validate (WITH CHECK) clause "
                "tied to app.tenant_id"
            )
        if reasons:
            violations[row.relname] = reasons
    return violations


async def test_public_schema_tenant_isolation_invariant(database_urls):
    """Acceptance: every public-schema table outside TENANT_ISOLATION_EXCEPTIONS carries
    forced RLS, a tenant column, and a policy with both a restrict and a validate clause."""
    from app.db.models import TENANT_ISOLATION_EXCEPTIONS

    engine = create_async_engine(database_urls["migrations"])
    async with engine.connect() as conn:
        violations = await _public_schema_tenant_isolation_violations(
            conn, TENANT_ISOLATION_EXCEPTIONS
        )
    await engine.dispose()
    assert violations == {}


async def test_alembic_bookkeeping_table_is_the_only_exception(database_urls):
    """Acceptance: the migration tool's own bookkeeping table is the only table on the
    exception list, and it genuinely needs the exemption (it really does carry none of the
    three mandatory parts) -- proving the list isn't hiding an under-protected domain table."""
    from app.db.models import TENANT_ISOLATION_EXCEPTIONS

    assert TENANT_ISOLATION_EXCEPTIONS == frozenset({"alembic_version"})

    engine = create_async_engine(database_urls["migrations"])
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT relrowsecurity FROM pg_class c "
                    "JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "WHERE n.nspname = 'public' AND c.relname = 'alembic_version'"
                )
            )
        ).one()
    await engine.dispose()
    assert row.relrowsecurity is False


async def test_invariant_catches_an_unprotected_table(database_urls):
    """Acceptance: adding a table with no policy to the schema under test causes the invariant
    check to fail -- proving it walks live catalog state rather than a fixed list of names."""
    from app.db.models import TENANT_ISOLATION_EXCEPTIONS

    engine = create_async_engine(database_urls["migrations"])
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE TABLE scratch_unprotected (id int)"))
        async with engine.connect() as conn:
            violations = await _public_schema_tenant_isolation_violations(
                conn, TENANT_ISOLATION_EXCEPTIONS
            )
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DROP TABLE IF EXISTS scratch_unprotected"))
        await engine.dispose()

    assert "scratch_unprotected" in violations
    assert "row security not enabled" in violations["scratch_unprotected"]


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


async def test_tenant_auth_settings_does_not_switch_the_callers_tenant_context(
    app_settings, database_urls
):
    """control.tenant_auth_settings() sets app.tenant_id internally to satisfy the forced RLS
    policy on control.tenants. It must restore the caller's context before returning —
    otherwise calling it for tenant B inside tenant A's session would leave the rest of that
    transaction running as tenant B."""
    from app.context import RequestContext
    from app.db.session import tenant_session
    from app.repositories.documents import DocumentRepository

    tenant_a, tenant_b = await _seed(database_urls["superuser"])
    query = [0.0] * DIM
    query[0] = 1.0

    ctx_a = RequestContext(tenant_id=tenant_a, identity_id=uuid.uuid4())
    async with tenant_session(ctx_a) as session:
        await session.execute(
            text("SELECT * FROM control.tenant_auth_settings(:tid)"), {"tid": tenant_b}
        )
        still_a = (
            await session.execute(text("SELECT current_setting('app.tenant_id', true)"))
        ).scalar_one()
        hits = await DocumentRepository().search(session, query, limit=10)
    assert still_a == str(tenant_a)
    assert [h.title for h in hits] == ["Document A"]


async def test_control_tenants_has_suspension_columns_defaulting_unsuspended(database_urls):
    """`control.tenants` gains a suspension flag and timestamp (#66), both defaulting to
    not-suspended so every existing and newly inserted tenant starts active."""
    superuser_engine = create_async_engine(database_urls["superuser"])
    tenant_id = uuid.uuid4()
    async with superuser_engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, 'A')"), {"id": tenant_id}
        )
        await conn.execute(
            text("INSERT INTO control.tenants (tenant_id) VALUES (:id)"), {"id": tenant_id}
        )
        row = (
            await conn.execute(
                text("SELECT suspended, suspended_at FROM control.tenants WHERE tenant_id = :id"),
                {"id": tenant_id},
            )
        ).one()
    await superuser_engine.dispose()
    assert row.suspended is False
    assert row.suspended_at is None


async def test_tenant_erasure_record_has_no_fk_and_survives_tenant_deletion(database_urls):
    """The erasure record (#66) is keyed by `tenant_id` with no foreign key back to the tenant
    it describes, so its row survives after the tenant row is deleted -- verified two ways:
    the column carries no FK constraint, and a real delete leaves the erasure row in place."""
    superuser_engine = create_async_engine(database_urls["superuser"])
    tenant_id = uuid.uuid4()
    async with superuser_engine.begin() as conn:
        fk_count = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM information_schema.table_constraints "
                    "WHERE table_schema = 'control' AND table_name = 'tenant_erasures' "
                    "AND constraint_type = 'FOREIGN KEY'"
                )
            )
        ).scalar_one()
        assert fk_count == 0

        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, 'A')"), {"id": tenant_id}
        )
        await conn.execute(
            text(
                "INSERT INTO control.tenant_erasures (tenant_id, details) "
                'VALUES (:id, \'{"removed": ["documents"]}\'::jsonb)'
            ),
            {"id": tenant_id},
        )
        await conn.execute(text("DELETE FROM tenants WHERE id = :id"), {"id": tenant_id})
        row = (
            await conn.execute(
                text("SELECT tenant_id FROM control.tenant_erasures WHERE tenant_id = :id"),
                {"id": tenant_id},
            )
        ).one()
    await superuser_engine.dispose()
    assert row.tenant_id == tenant_id


async def test_operator_actions_table_also_has_no_fk_to_tenants(database_urls):
    """Same reasoning as the erasure record: an `erase` action's own log entry must outlive the
    tenant it names, so `control.operator_actions` carries no foreign key either (#66)."""
    superuser_engine = create_async_engine(database_urls["superuser"])
    async with superuser_engine.begin() as conn:
        fk_count = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM information_schema.table_constraints "
                    "WHERE table_schema = 'control' AND table_name = 'operator_actions' "
                    "AND constraint_type = 'FOREIGN KEY'"
                )
            )
        ).scalar_one()
    await superuser_engine.dispose()
    assert fk_count == 0


@pytest.mark.parametrize("table", ["tenant_erasures", "operator_actions"])
async def test_append_only_tables_accept_insert_reject_update_delete_by_owner(database_urls, table):
    """Both new tables (#66) accept INSERT from `app_owner`, which owns them, but reject UPDATE
    and DELETE from that same role: table ownership grants privileges exactly as if by GRANT,
    and migration 0004 explicitly revokes everything but INSERT from `app_owner` -- append-only
    applies to the owner too, not just to unprivileged roles."""
    from sqlalchemy.exc import DBAPIError

    superuser_engine = create_async_engine(database_urls["superuser"])
    tenant_id = uuid.uuid4()
    async with superuser_engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, 'A')"), {"id": tenant_id}
        )
    await superuser_engine.dispose()

    owner_engine = create_async_engine(database_urls["migrations"])
    if table == "tenant_erasures":
        insert_sql = "INSERT INTO control.tenant_erasures (tenant_id) VALUES (:id)"
    else:
        insert_sql = (
            "INSERT INTO control.operator_actions (tenant_id, action) VALUES (:id, 'create')"
        )

    async with owner_engine.begin() as conn:
        await conn.execute(text(insert_sql), {"id": tenant_id})

    with pytest.raises(DBAPIError):
        async with owner_engine.begin() as conn:
            await conn.execute(
                text(f"UPDATE control.{table} SET tenant_id = tenant_id WHERE tenant_id = :id"),
                {"id": tenant_id},
            )

    with pytest.raises(DBAPIError):
        async with owner_engine.begin() as conn:
            await conn.execute(
                text(f"DELETE FROM control.{table} WHERE tenant_id = :id"), {"id": tenant_id}
            )

    # Owner also lost SELECT: only INSERT remains (ticket #66 grants only that).
    with pytest.raises(DBAPIError):
        async with owner_engine.begin() as conn:
            await conn.execute(
                text(f"SELECT * FROM control.{table} WHERE tenant_id = :id"), {"id": tenant_id}
            )

    await owner_engine.dispose()


@pytest.mark.parametrize("table", ["tenant_erasures", "operator_actions"])
async def test_select_on_append_only_tables_is_grantable_to_an_auditor_role(database_urls, table):
    """SELECT on either table is grantable to a distinct role without handing that role any
    write access (#66) -- proving the two privileges are independent, not bundled."""
    from sqlalchemy.exc import DBAPIError

    role = f"test_auditor_{table}"
    superuser_engine = create_async_engine(database_urls["superuser"])
    tenant_id = uuid.uuid4()
    try:
        async with superuser_engine.begin() as conn:
            await conn.execute(text(f"CREATE ROLE {role} LOGIN NOSUPERUSER NOBYPASSRLS"))
            await conn.execute(
                text("INSERT INTO tenants (id, name) VALUES (:id, 'A')"), {"id": tenant_id}
            )

        owner_engine = create_async_engine(database_urls["migrations"])
        insert_sql = (
            f"INSERT INTO control.{table} (tenant_id) VALUES (:id)"
            if table == "tenant_erasures"
            else f"INSERT INTO control.{table} (tenant_id, action) VALUES (:id, 'create')"
        )
        async with owner_engine.begin() as conn:
            await conn.execute(text(insert_sql), {"id": tenant_id})
            await conn.execute(text(f"GRANT USAGE ON SCHEMA control TO {role}"))
            await conn.execute(text(f"GRANT SELECT ON control.{table} TO {role}"))
        await owner_engine.dispose()

        parsed = urlparse(database_urls["app"])
        sockdir = parse_qs(parsed.query)["host"][0]
        auditor_engine = create_async_engine(
            f"postgresql+asyncpg://{role}@/postgres?host={sockdir}"
        )
        async with auditor_engine.connect() as conn:
            row = (
                await conn.execute(
                    text(f"SELECT tenant_id FROM control.{table} WHERE tenant_id = :id"),
                    {"id": tenant_id},
                )
            ).one()
            assert row.tenant_id == tenant_id

            with pytest.raises(DBAPIError):
                await conn.execute(
                    text(f"DELETE FROM control.{table} WHERE tenant_id = :id"), {"id": tenant_id}
                )
        await auditor_engine.dispose()
    finally:
        cleanup_engine = create_async_engine(database_urls["superuser"])
        async with cleanup_engine.begin() as conn:
            await conn.execute(text("DELETE FROM tenants WHERE id = :id"), {"id": tenant_id})
            await conn.execute(text(f"REVOKE ALL ON control.{table} FROM {role}"))
            await conn.execute(text(f"REVOKE ALL ON SCHEMA control FROM {role}"))
            await conn.execute(text(f"DROP ROLE {role}"))
        await cleanup_engine.dispose()


async def _insert_public_tenant(conn, tenant_id: uuid.UUID, name: str) -> None:
    await conn.execute(
        text("INSERT INTO tenants (id, name) VALUES (:id, :name)"), {"id": tenant_id, "name": name}
    )


async def _insert_control_tenant(
    conn, tenant_id: uuid.UUID, *, isolation_tier: str, database_alias: str | None
) -> None:
    await conn.execute(
        text(
            "INSERT INTO control.tenants (tenant_id, isolation_tier, database_alias) "
            "VALUES (:tid, :tier, :alias)"
        ),
        {"tid": tenant_id, "tier": isolation_tier, "alias": database_alias},
    )


async def test_pooled_tier_with_alias_rejected_by_schema_constraint(database_urls):
    """(#73) A pooled tenant's database alias must be NULL -- enforced by the `CHECK`
    constraint added in 0005, not by application code."""
    from sqlalchemy.exc import DBAPIError

    engine = create_async_engine(database_urls["superuser"])
    tenant_id = uuid.uuid4()
    async with engine.begin() as conn:
        await _insert_public_tenant(conn, tenant_id, "pooled-with-alias")
    with pytest.raises(DBAPIError):
        async with engine.begin() as conn:
            await _insert_control_tenant(
                conn, tenant_id, isolation_tier="pooled", database_alias="eu1"
            )
    await engine.dispose()


async def test_dedicated_tier_without_alias_rejected_by_schema_constraint(database_urls):
    """(#73) A dedicated tenant's database alias must be set -- enforced the same way."""
    from sqlalchemy.exc import DBAPIError

    engine = create_async_engine(database_urls["superuser"])
    tenant_id = uuid.uuid4()
    async with engine.begin() as conn:
        await _insert_public_tenant(conn, tenant_id, "dedicated-without-alias")
    with pytest.raises(DBAPIError):
        async with engine.begin() as conn:
            await _insert_control_tenant(
                conn, tenant_id, isolation_tier="dedicated", database_alias=None
            )
    await engine.dispose()


async def test_database_aliases_view_enumerates_distinct_aliases(database_urls):
    """(#73) `control.database_aliases` returns exactly the pooled default while every tenant
    is pooled, and includes a newly seeded alias once a dedicated tenant is added."""
    engine = create_async_engine(database_urls["superuser"])
    pooled_a, pooled_b, dedicated = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    async with engine.begin() as conn:
        for tenant_id, name in ((pooled_a, "pooled-a"), (pooled_b, "pooled-b")):
            await _insert_public_tenant(conn, tenant_id, name)
            await _insert_control_tenant(
                conn, tenant_id, isolation_tier="pooled", database_alias=None
            )

    async with engine.connect() as conn:
        aliases = set(
            (await conn.execute(text("SELECT database_alias FROM control.database_aliases")))
            .scalars()
            .all()
        )
    assert aliases == {"pooled"}

    async with engine.begin() as conn:
        await _insert_public_tenant(conn, dedicated, "dedicated-eu1")
        await _insert_control_tenant(
            conn, dedicated, isolation_tier="dedicated", database_alias="eu1"
        )

    async with engine.connect() as conn:
        aliases = set(
            (await conn.execute(text("SELECT database_alias FROM control.database_aliases")))
            .scalars()
            .all()
        )
    await engine.dispose()
    assert aliases == {"pooled", "eu1"}


async def test_control_tenants_stores_only_the_alias_string(database_urls):
    """(#73) The dedicated-tenant record has exactly one column for "where its database
    lives" -- the alias string -- and no column for a DSN, hostname, port, user, or password:
    that connection detail lives in a tenant secret file keyed by the alias (ADR-0011), never
    in the control plane's own schema."""
    engine = create_async_engine(database_urls["migrations"])
    async with engine.connect() as conn:
        columns = (
            await conn.execute(
                text(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_schema = 'control' AND table_name = 'tenants'"
                )
            )
        ).all()
    await engine.dispose()
    by_name = {row.column_name: row.data_type for row in columns}
    assert by_name["database_alias"] == "text"
    connection_detail = ("dsn", "url", "host", "port", "user", "password")
    offending = {c for c in by_name if any(word in c for word in connection_detail)}
    assert offending == set()


async def test_tenants_view_still_select_only_after_adding_isolation_columns(database_urls):
    """(#73) Extending `control.tenants_view` with `isolation_tier`/`database_alias` adds no
    new grant: `app` still holds SELECT only, exactly as after #12."""
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
        columns = (
            (
                await conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns WHERE table_schema = "
                        "'control' AND table_name = 'tenants_view'"
                    )
                )
            )
            .scalars()
            .all()
        )
    await engine.dispose()
    # identity_lookup (0003) is the only other object app may read; nothing but SELECT anywhere.
    # (The runtime guard enumerates aliases via EXECUTE on a function, 0017 / #77.)
    assert set(table_grants) == {
        ("tenants_view", "SELECT"),
        ("identity_lookup", "SELECT"),
    }
    assert {"isolation_tier", "database_alias"} <= set(columns)


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


# --- Live role/RLS guard (issue #15 / ADR-0011): the same exception list as the ---
# --- schema-invariant test above, exercised through the actual runtime check. ---


async def test_role_rls_guard_passes_for_the_unprivileged_app_role(database_urls):
    """Acceptance: the live guard does not raise for a correctly configured, unprivileged
    connection — proving the earlier failure tests below are not tautologies."""
    from app.db.guard import check_role_and_rls

    engine = create_async_engine(database_urls["app"])
    async with engine.connect() as conn:
        await check_role_and_rls(conn)  # must not raise
    await engine.dispose()


async def test_role_rls_guard_raises_for_a_superuser_or_bypassrls_role(database_urls):
    """Acceptance: the live check raises when connected as a role that is a superuser or holds
    BYPASSRLS — pgserver's own bootstrap `postgres` role, mirroring the cluster's real
    bootstrap superuser."""
    from app.db.guard import PrivilegedRoleOrMissingRLSError, check_role_and_rls

    engine = create_async_engine(database_urls["superuser"])
    async with engine.connect() as conn:
        with pytest.raises(PrivilegedRoleOrMissingRLSError):
            await check_role_and_rls(conn)
    await engine.dispose()


async def test_role_rls_guard_raises_for_a_table_missing_forced_rls(database_urls):
    """Acceptance: a table in `public` outside `TENANT_ISOLATION_EXCEPTIONS` that lacks forced
    Row-Level Security makes the live guard raise, even when connected as the unprivileged
    `app` role — the same exception list the schema-invariant test enforces, checked here
    through the actual runtime guard rather than the test-only catalog walk."""
    from app.db.guard import PrivilegedRoleOrMissingRLSError, check_role_and_rls
    from app.db.models import TENANT_ISOLATION_EXCEPTIONS

    assert "scratch_unforced" not in TENANT_ISOLATION_EXCEPTIONS
    owner_engine = create_async_engine(database_urls["migrations"])
    try:
        async with owner_engine.begin() as conn:
            await conn.execute(text("CREATE TABLE scratch_unforced (id int)"))

        app_engine = create_async_engine(database_urls["app"])
        try:
            async with app_engine.connect() as conn:
                with pytest.raises(PrivilegedRoleOrMissingRLSError):
                    await check_role_and_rls(conn)
        finally:
            await app_engine.dispose()
    finally:
        async with owner_engine.begin() as conn:
            await conn.execute(text("DROP TABLE IF EXISTS scratch_unforced"))
        await owner_engine.dispose()


async def test_ready_endpoint_succeeds_against_real_postgres_as_the_app_role(
    app_settings, database_urls
):
    """ASGI seam, real backing Postgres (issue #15): the readiness endpoint's default
    dependency — left as-is, not overridden with a fake — succeeds against a real,
    correctly-configured connection as the unprivileged `app` role. The guard (#77) now also
    enumerates dedicated aliases; this module-scoped instance is shared with earlier tests that
    deliberately seed a dedicated tenant with no matching secret file
    (`test_database_aliases_view_enumerates_distinct_aliases`), so that alias is cleared here
    first -- this test is only about the pooled path succeeding, not about a dangling alias
    another test left behind."""
    import httpx

    from app.main import app as main_app

    superuser_engine = create_async_engine(database_urls["superuser"])
    async with superuser_engine.begin() as conn:
        await conn.execute(
            text("UPDATE control.tenants SET isolation_tier = 'pooled', database_alias = NULL")
        )
    await superuser_engine.dispose()

    transport = httpx.ASGITransport(app=main_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


# --- Residency route resolution (#60, ADR-0008): the resolver end to end, real RLS ---------


async def _set_residency(url: str, tenant_id: uuid.UUID, residency: str | None) -> None:
    """As the superuser, exactly like `_seed` above: writes `tenants.settings["residency"]`
    directly rather than through the resolver-under-test, so the resolver's own read is what's
    being verified, not a round trip through itself."""
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE tenants SET settings = jsonb_set("
                "coalesce(settings, '{}'::jsonb), '{residency}', "
                "to_jsonb(CAST(:residency AS text))) "
                "WHERE id = :tid"
            ),
            {"tid": tenant_id, "residency": residency},
        )
    await engine.dispose()


async def test_residency_resolves_through_real_rls_and_never_crosses_tenants(
    app_settings, database_urls, tmp_path
):
    """Given two tenants set to different residencies, resolving one tenant's route never
    returns the other's -- read through the same tenant-scoped session/RLS scaffolding every
    repository uses, exactly like the document-search isolation test above (#60)."""
    from app.config import RESIDENCY_ALLOW_LIST, Settings
    from app.context import RequestContext
    from app.db.session import tenant_session
    from app.residency import resolve_residency_route

    tenant_eu, tenant_us = await _seed(database_urls["superuser"])
    await _set_residency(database_urls["superuser"], tenant_eu, "eu")
    await _set_residency(database_urls["superuser"], tenant_us, "us")
    await _set_gateway_alias(database_urls["migrations"], tenant_eu, "acme-gateway-key")
    await _set_gateway_alias(database_urls["migrations"], tenant_us, "globex-gateway-key")
    (tmp_path / "acme-gateway-key").write_text("sk-acme-secret")
    (tmp_path / "globex-gateway-key").write_text("sk-globex-secret")
    settings = Settings(database_url=database_urls["app"], gateway_credentials_dir=str(tmp_path))

    ctx_eu = RequestContext(tenant_id=tenant_eu, identity_id=uuid.uuid4())
    async with tenant_session(ctx_eu) as session:
        resolved_eu = await resolve_residency_route(session, ctx_eu, settings=settings)

    ctx_us = RequestContext(tenant_id=tenant_us, identity_id=uuid.uuid4())
    async with tenant_session(ctx_us) as session:
        resolved_us = await resolve_residency_route(session, ctx_us, settings=settings)

    assert resolved_eu.residency == "eu"
    assert resolved_eu.route == RESIDENCY_ALLOW_LIST["eu"]
    assert resolved_eu.gateway_credential.get_secret_value() == "sk-acme-secret"

    assert resolved_us.residency == "us"
    assert resolved_us.route == RESIDENCY_ALLOW_LIST["us"]
    assert resolved_us.gateway_credential.get_secret_value() == "sk-globex-secret"

    # Never each other's route or credential.
    assert resolved_eu.route != resolved_us.route
    assert resolved_eu.route.trace_sink_host != resolved_us.route.trace_sink_host
    assert (
        resolved_eu.gateway_credential.get_secret_value()
        != resolved_us.gateway_credential.get_secret_value()
    )


async def test_residency_resolution_is_scoped_by_rls_not_just_the_where_clause(
    app_settings, database_urls, tmp_path
):
    """Even if the resolver's own WHERE clause were removed or broken, RLS on `tenants` (0001)
    would still stop tenant A's session from ever reading tenant B's row: `tenant_session(ctx)`
    sets `app.tenant_id` per transaction, and the `tenants_self_only` policy filters on it."""
    from app.context import RequestContext
    from app.db.session import tenant_session

    tenant_eu, tenant_us = await _seed(database_urls["superuser"])
    await _set_residency(database_urls["superuser"], tenant_eu, "eu")
    await _set_residency(database_urls["superuser"], tenant_us, "us")

    ctx_eu = RequestContext(tenant_id=tenant_eu, identity_id=uuid.uuid4())
    async with tenant_session(ctx_eu) as session:
        # Querying tenant B's row from tenant A's session-scoped connection returns nothing:
        # RLS filters it out before this WHERE clause is even considered.
        row = (
            await session.execute(
                text("SELECT settings ->> 'residency' FROM tenants WHERE id = :tid"),
                {"tid": tenant_us},
            )
        ).first()
    assert row is None


async def test_resolver_fails_closed_for_a_tenant_with_no_residency_set(
    app_settings, database_urls
):
    from app.context import RequestContext
    from app.db.session import tenant_session
    from app.residency import ResidencyUnresolved, resolve_residency_route

    tenant_a, _ = await _seed(database_urls["superuser"])
    ctx_a = RequestContext(tenant_id=tenant_a, identity_id=uuid.uuid4())
    with pytest.raises(ResidencyUnresolved):
        async with tenant_session(ctx_a) as session:
            await resolve_residency_route(session, ctx_a)


async def test_resolver_fails_closed_for_a_tenant_with_an_unknown_residency(
    app_settings, database_urls
):
    from app.context import RequestContext
    from app.db.session import tenant_session
    from app.residency import ResidencyUnresolved, resolve_residency_route

    tenant_a, _ = await _seed(database_urls["superuser"])
    await _set_residency(database_urls["superuser"], tenant_a, "mars")
    ctx_a = RequestContext(tenant_id=tenant_a, identity_id=uuid.uuid4())
    with pytest.raises(ResidencyUnresolved):
        async with tenant_session(ctx_a) as session:
            await resolve_residency_route(session, ctx_a)


# --- Tenant memberships replace per-tenant users (ADR-0003, Spec 2 / #23) ---


async def _seed_identity(url: str, *, subject: str) -> uuid.UUID:
    """A global identity, inserted with the superuser exactly like `_seed`'s tenants/documents:
    `control.identities` carries no tenant_id and no RLS (ADR-0003)."""
    engine = create_async_engine(url)
    identity_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO control.identities (id, issuer, subject) "
                "VALUES (:id, 'https://idp.example.com', :subject)"
            ),
            {"id": identity_id, "subject": subject},
        )
    await engine.dispose()
    return identity_id


async def _seed_membership(
    url: str, *, tenant_id: uuid.UUID, identity_id: uuid.UUID, role: str
) -> None:
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO memberships (tenant_id, identity_id, role) "
                "VALUES (:tenant_id, :identity_id, :role)"
            ),
            {"tenant_id": tenant_id, "identity_id": identity_id, "role": role},
        )
    await engine.dispose()


async def test_migration_drops_users_and_memberships_has_the_expected_shape(database_urls):
    """Acceptance: migrating drops the old per-tenant `users` table and creates `memberships`
    with a required indexed tenant reference, a cross-schema reference to `control.identities`,
    a role restricted to the four defined roles, one membership per identity per tenant, and RLS
    enabled and forced."""
    engine = create_async_engine(database_urls["migrations"])
    async with engine.connect() as conn:
        users_exists = (
            await conn.execute(
                text(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name = 'users')"
                )
            )
        ).scalar_one()
        assert users_exists is False

        rls = (
            await conn.execute(
                text(
                    "SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
                    "WHERE oid = CAST('memberships' AS regclass)"
                )
            )
        ).one()
        assert rls.relrowsecurity is True
        assert rls.relforcerowsecurity is True

        tenant_idx = (
            await conn.execute(
                text(
                    "SELECT indexdef FROM pg_indexes "
                    "WHERE schemaname = 'public' AND tablename = 'memberships' "
                    "AND indexname = 'memberships_tenant_idx'"
                )
            )
        ).scalar_one_or_none()
        assert tenant_idx is not None

        not_null_tenant = (
            await conn.execute(
                text(
                    "SELECT is_nullable FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = 'memberships' "
                    "AND column_name = 'tenant_id'"
                )
            )
        ).scalar_one()
        assert not_null_tenant == "NO"

        identity_fk_target = (
            await conn.execute(
                text(
                    """
                    SELECT ccu.table_schema, ccu.table_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.constraint_column_usage ccu
                        ON tc.constraint_name = ccu.constraint_name
                        AND tc.constraint_schema = ccu.constraint_schema
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.constraint_schema = kcu.constraint_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                        AND tc.table_schema = 'public' AND tc.table_name = 'memberships'
                        AND kcu.column_name = 'identity_id'
                    """
                )
            )
        ).one()
    await engine.dispose()
    assert (identity_fk_target.table_schema, identity_fk_target.table_name) == (
        "control",
        "identities",
    )

    # One membership per identity per tenant.
    tenant_id = uuid.uuid4()
    identity_id = await _seed_identity(database_urls["superuser"], subject="s-unique")
    superuser_engine = create_async_engine(database_urls["superuser"])
    async with superuser_engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, 'Uniq')"), {"id": tenant_id}
        )
    await superuser_engine.dispose()
    await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, identity_id=identity_id, role="member"
    )
    from sqlalchemy.exc import DBAPIError, IntegrityError

    with pytest.raises((DBAPIError, IntegrityError)):
        await _seed_membership(
            database_urls["superuser"], tenant_id=tenant_id, identity_id=identity_id, role="admin"
        )

    # Role restricted to the four defined roles.
    other_identity_id = await _seed_identity(database_urls["superuser"], subject="s-bad-role")
    with pytest.raises((DBAPIError, IntegrityError)):
        await _seed_membership(
            database_urls["superuser"],
            tenant_id=tenant_id,
            identity_id=other_identity_id,
            role="owner",
        )


async def test_memberships_grants_mirror_the_retired_users_grants(database_urls):
    """Acceptance: grants on `memberships` for `app` mirror the retired `users` grants."""
    engine = create_async_engine(database_urls["app"])
    async with engine.connect() as conn:
        privileges = (
            (
                await conn.execute(
                    text(
                        "SELECT privilege_type FROM information_schema.table_privileges "
                        "WHERE table_schema = 'public' AND table_name = 'memberships' "
                        "AND grantee = 'app'"
                    )
                )
            )
            .scalars()
            .all()
        )
    await engine.dispose()
    assert set(privileges) == {"SELECT", "INSERT", "UPDATE", "DELETE"}


async def test_second_tenants_memberships_are_invisible_without_its_own_context(
    app_settings, database_urls
):
    """Acceptance (mirrors the retired users coverage): a second tenant's membership rows are
    invisible without that tenant's own context set, and a cross-tenant insert or update fails
    the write check."""
    from sqlalchemy.exc import DBAPIError

    from app.context import RequestContext
    from app.db.session import tenant_session

    tenant_a, tenant_b = await _seed(database_urls["superuser"])
    identity_a = await _seed_identity(database_urls["superuser"], subject="s-a")
    identity_b = await _seed_identity(database_urls["superuser"], subject="s-b")
    await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_a, identity_id=identity_a, role="admin"
    )
    await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_b, identity_id=identity_b, role="admin"
    )

    ctx_a = RequestContext(tenant_id=tenant_a, identity_id=identity_a)
    async with tenant_session(ctx_a) as session:
        rows = (await session.execute(text("SELECT tenant_id FROM memberships"))).scalars().all()
    assert rows == [tenant_a]

    # Cross-tenant insert is rejected by the WITH CHECK clause.
    with pytest.raises(DBAPIError):
        async with tenant_session(ctx_a) as session:
            await session.execute(
                text(
                    "INSERT INTO memberships (tenant_id, identity_id, role) "
                    "VALUES (:tid, :iid, 'member')"
                ),
                {"tid": tenant_b, "iid": identity_b},
            )

    # Cross-tenant update (targeting the other tenant's row) is rejected: the USING clause
    # hides the row from tenant A's session in the first place, so zero rows are affected.
    async with tenant_session(ctx_a) as session:
        result = await session.execute(
            text("UPDATE memberships SET role = 'support' WHERE tenant_id = :tid"),
            {"tid": tenant_b},
        )
        assert result.rowcount == 0


async def test_membership_lookup_returns_role_or_none(app_settings, database_urls):
    """Acceptance: the membership lookup returns the role for an existing tenant/identity pair
    and returns nothing rather than raising when no row matches."""
    from app.context import RequestContext
    from app.db.session import tenant_session
    from app.repositories.memberships import MembershipRepository

    tenant_a, _ = await _seed(database_urls["superuser"])
    identity_a = await _seed_identity(database_urls["superuser"], subject="s-lookup")
    await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_a, identity_id=identity_a, role="support"
    )

    ctx_a = RequestContext(tenant_id=tenant_a, identity_id=identity_a)
    async with tenant_session(ctx_a) as session:
        found = await MembershipRepository().get_role(session, ctx_a, identity_id=identity_a)
        missing = await MembershipRepository().get_role(session, ctx_a, identity_id=uuid.uuid4())
    assert found == "support"
    assert missing is None


async def test_list_for_tenant_returns_every_role_unfiltered(app_settings, database_urls):
    """Acceptance: listing a tenant's memberships returns every row including a support-role
    one, with no role-based filtering."""
    from app.context import RequestContext
    from app.db.session import tenant_session
    from app.repositories.memberships import MembershipRepository

    tenant_a, _ = await _seed(database_urls["superuser"])
    identities = {
        role: await _seed_identity(database_urls["superuser"], subject=f"s-{role}")
        for role in ("admin", "member", "support", "agent")
    }
    for role, identity_id in identities.items():
        await _seed_membership(
            database_urls["superuser"], tenant_id=tenant_a, identity_id=identity_id, role=role
        )

    ctx_a = RequestContext(tenant_id=tenant_a, identity_id=identities["admin"])
    async with tenant_session(ctx_a) as session:
        records = await MembershipRepository().list_for_tenant(session, ctx_a)

    assert {record.role for record in records} == {"admin", "member", "support", "agent"}
    assert len(records) == 4


async def test_app_cannot_widen_its_control_plane_view_with_the_migration_read_flag(
    database_urls,
):
    """0016's migration-read policy exists for the owner-role migration runner only. `app` can
    set any custom setting itself, so setting `app.control_migration_read` must still leave it
    seeing nothing but its own tenant's control-plane row."""
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    owner = create_async_engine(database_urls["superuser"])
    async with owner.begin() as conn:
        for tid, name in ((tenant_a, "Flag A"), (tenant_b, "Flag B")):
            await _insert_public_tenant(conn, tid, name)
            await _insert_control_tenant(conn, tid, isolation_tier="pooled", database_alias=None)
    await owner.dispose()

    app = create_async_engine(database_urls["app"])
    async with app.begin() as conn:
        await conn.execute(text("SELECT set_config('app.control_migration_read', 'true', true)"))
        await conn.execute(
            text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": str(tenant_a)}
        )
        visible = (
            (await conn.execute(text("SELECT tenant_id FROM control.tenants_view"))).scalars().all()
        )
    await app.dispose()
    assert visible == [tenant_a]


async def test_app_without_tenant_context_sees_no_control_plane_rows(database_urls):
    """No context means no rows -- for the control plane too. The guard's cross-tenant alias
    enumeration goes through `control.enumerate_database_aliases()` only, never by widening what
    a context-less `app` session can read from `control.tenants_view`."""
    tenant_id = uuid.uuid4()
    owner = create_async_engine(database_urls["superuser"])
    async with owner.begin() as conn:
        await _insert_public_tenant(conn, tenant_id, "No-context")
        await _insert_control_tenant(
            conn, tenant_id, isolation_tier="dedicated", database_alias="no-context-alias"
        )
    await owner.dispose()

    app = create_async_engine(database_urls["app"])
    async with app.begin() as conn:
        visible = (await conn.execute(text("SELECT tenant_id FROM control.tenants_view"))).all()
        aliases = (
            (await conn.execute(text("SELECT * FROM control.enumerate_database_aliases()")))
            .scalars()
            .all()
        )
        after = (await conn.execute(text("SELECT tenant_id FROM control.tenants_view"))).all()
    await app.dispose()
    assert visible == [] and after == []
    assert "no-context-alias" in aliases
