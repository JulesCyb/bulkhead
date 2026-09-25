"""Embedded-Postgres integration tests for `tenant_session()` routing through the control plane
and the engine registry (ADR-0002, Spec 10 / #75).

Two ephemeral `pgserver` instances stand in for the pooled default database and a dedicated
tenant's own database. Both get the full migration suite (so `control.tenants`, `tenants`, and
RLS all exist for real). A dedicated tenant's control-plane bookkeeping row necessarily lives in
the pooled database too (`control.tenants` FK-references `public.tenants`) -- what these tests
prove absent from the pooled database is the tenant's own *data* (a row in `memberships`), not that
bookkeeping stub.
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

pgserver = pytest.importorskip("pgserver")


def _psql(server, command: str) -> None:
    """`server.psql` without a shell: pgserver's own version breaks on paths with spaces."""
    from pgserver.postgres_server import POSTGRES_BIN_PATH

    subprocess.run(
        [str(POSTGRES_BIN_PATH / "psql"), server.get_uri()],
        input=command.encode(),
        check=True,
        capture_output=True,
    )


def _bootstrap_and_migrate(server) -> dict[str, str]:
    """Mirrors docker/postgres/01-init.sh + `alembic upgrade head` (see test_rls_integration.py)
    against one ephemeral instance, returning its owner/app connection strings."""
    from app.config import ROLE_STATEMENT_TIMEOUT_MS

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
        "migrations": f"postgresql+asyncpg://app_owner@/postgres?host={sockdir}",
        "app": f"postgresql+asyncpg://app@/postgres?host={sockdir}",
    }
    env = {**os.environ, "DATABASE_URL_MIGRATIONS": urls["migrations"], "DATABASE_URL": urls["app"]}
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"], check=True, env=env, timeout=120
    )
    return urls


@pytest.fixture(scope="module")
def two_databases():
    """Index 0: the pooled default. Index 1: a dedicated tenant's own database. Both fully
    migrated, both with the real `app_owner`/`app` roles."""
    servers = []
    urls = []
    for _ in range(2):
        pgdata = tempfile.mkdtemp(prefix="pgdata-tenant-session-routing-")
        server = pgserver.get_server(pgdata, cleanup_mode="delete")
        urls.append(_bootstrap_and_migrate(server))
        servers.append(server)
    yield urls
    for server in servers:
        server.cleanup()


@pytest.fixture
def routing_env(two_databases, tmp_path, monkeypatch):
    from app import config
    from app.db import engine_registry
    from app.db import session as db_session

    pooled_urls, dedicated_urls = two_databases

    monkeypatch.setenv("DATABASE_URL", pooled_urls["app"])
    monkeypatch.setenv("DATABASE_URL_MIGRATIONS", pooled_urls["migrations"])
    config.get_settings.cache_clear()
    db_session._engine = None
    db_session._session_factory = None
    engine_registry.reset_registry_for_tests()

    secrets_dir = tmp_path / "tenant-db"
    secrets_dir.mkdir()
    monkeypatch.setenv("TENANT_DB_SECRETS_DIR", str(secrets_dir))
    (secrets_dir / "tenant-dedicated").write_text(dedicated_urls["app"])

    yield {"pooled": pooled_urls, "dedicated": dedicated_urls, "secrets_dir": secrets_dir}

    config.get_settings.cache_clear()
    db_session._engine = None
    db_session._session_factory = None
    engine_registry.reset_registry_for_tests()


async def _seed_tenant(
    migrations_url: str,
    tenant_id: uuid.UUID,
    name: str,
    *,
    control_row: tuple[str, str | None] | None,
) -> None:
    """Writes the tenant's `public.tenants` row and, if given, its `control.tenants` bookkeeping
    row, as `app_owner`. `app_owner` is not a superuser and does not bypass RLS (FORCE ROW LEVEL
    SECURITY applies to it too), so the tenant context must be set before either insert -- exactly
    what a real owner-role operator action would do.
    """
    engine = create_async_engine(migrations_url)
    async with engine.begin() as conn:
        await conn.execute(
            text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": str(tenant_id)}
        )
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant_id, "name": name},
        )
        if control_row is not None:
            isolation_tier, database_alias = control_row
            await conn.execute(
                text(
                    "INSERT INTO control.tenants (tenant_id, isolation_tier, database_alias) "
                    "VALUES (:tid, :tier, :alias)"
                ),
                {"tid": tenant_id, "tier": isolation_tier, "alias": database_alias},
            )
    await engine.dispose()


async def _seed_user(migrations_url: str, tenant_id: uuid.UUID, email: str) -> None:
    """A global identity (subject = email) plus its membership in `tenant_id` -- the tenant's own
    row of data these tests look for."""
    engine = create_async_engine(migrations_url)
    identity_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO control.identities (id, issuer, subject) VALUES (:id, :iss, :sub)"),
            # A fresh issuer per call: the databases are module-scoped, the emails repeat.
            {"id": identity_id, "iss": f"test-{identity_id}", "sub": email},
        )
        await conn.execute(
            text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": str(tenant_id)}
        )
        await conn.execute(
            text(
                "INSERT INTO memberships (tenant_id, identity_id, role) "
                "VALUES (:tid, :iid, 'member')"
            ),
            {"tid": tenant_id, "iid": identity_id},
        )
    await engine.dispose()


_MEMBER_EMAILS = (
    "SELECT l.subject FROM memberships m JOIN control.identity_lookup l ON l.id = m.identity_id"
)


async def test_pooled_tenant_is_served_from_the_default_instance_and_sees_only_its_own_rows(
    routing_env,
):
    from app.context import RequestContext
    from app.db.session import get_engine, tenant_session

    pooled_urls = routing_env["pooled"]
    tenant = uuid.uuid4()
    other = uuid.uuid4()
    await _seed_tenant(pooled_urls["migrations"], tenant, "Pooled", control_row=("pooled", None))
    await _seed_tenant(pooled_urls["migrations"], other, "Other", control_row=("pooled", None))
    await _seed_user(pooled_urls["migrations"], tenant, "pooled@example.com")
    await _seed_user(pooled_urls["migrations"], other, "other@example.com")

    ctx = RequestContext(tenant_id=tenant, identity_id=uuid.uuid4())
    async with tenant_session(ctx) as session:
        assert session.get_bind() is get_engine().sync_engine
        emails = (await session.execute(text(_MEMBER_EMAILS))).scalars().all()
        assert emails == ["pooled@example.com"]


async def test_dedicated_tenant_is_served_from_its_own_instance_and_sees_only_its_own_rows(
    routing_env,
):
    from app.context import RequestContext
    from app.db.session import get_engine, tenant_session

    pooled_urls, dedicated_urls = routing_env["pooled"], routing_env["dedicated"]
    tenant = uuid.uuid4()
    # Control-plane bookkeeping: only ever in the pooled database.
    await _seed_tenant(
        pooled_urls["migrations"],
        tenant,
        "Dedicated",
        control_row=("dedicated", "tenant-dedicated"),
    )
    # The tenant's actual data lives on its own dedicated instance.
    await _seed_tenant(dedicated_urls["migrations"], tenant, "Dedicated", control_row=None)
    await _seed_user(dedicated_urls["migrations"], tenant, "dedicated@example.com")

    ctx = RequestContext(tenant_id=tenant, identity_id=uuid.uuid4())
    async with tenant_session(ctx) as session:
        assert session.get_bind() is not get_engine().sync_engine
        emails = (await session.execute(text(_MEMBER_EMAILS))).scalars().all()
        assert emails == ["dedicated@example.com"]


async def test_dedicated_tenants_data_is_physically_absent_from_the_pooled_database(routing_env):
    """Not merely policy-hidden: forcing the tenant's own context directly against the pooled
    engine (bypassing routing entirely) still returns zero rows, because the tenant's data was
    never written to the pooled database -- only its control-plane bookkeeping stub was."""
    from app.context import RequestContext
    from app.db.session import get_engine, tenant_session

    pooled_urls, dedicated_urls = routing_env["pooled"], routing_env["dedicated"]
    tenant = uuid.uuid4()
    await _seed_tenant(
        pooled_urls["migrations"],
        tenant,
        "Dedicated",
        control_row=("dedicated", "tenant-dedicated"),
    )
    await _seed_tenant(dedicated_urls["migrations"], tenant, "Dedicated", control_row=None)
    await _seed_user(dedicated_urls["migrations"], tenant, "dedicated@example.com")

    ctx = RequestContext(tenant_id=tenant, identity_id=uuid.uuid4())
    async with tenant_session(ctx):
        pass  # exercises routing once; asserted directly against tenant_session() above

    pooled_engine = get_engine()
    async with pooled_engine.connect() as conn:
        async with conn.begin():
            await conn.execute(
                text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": str(tenant)}
            )
            emails = (await conn.execute(text(_MEMBER_EMAILS))).scalars().all()
    assert emails == []


async def test_tenant_with_no_control_plane_row_defaults_to_pooled(routing_env):
    """Existing callers (test_rls_integration.py, test_db_limits_integration.py) seed tenants
    only in `public.tenants`, never in `control.tenants` -- ADR-0002 defaults every tenant to
    pooled, so this must keep working unmodified."""
    from app.context import RequestContext
    from app.db.session import get_engine, tenant_session

    pooled_urls = routing_env["pooled"]
    tenant = uuid.uuid4()
    await _seed_tenant(pooled_urls["migrations"], tenant, "NoControlRow", control_row=None)
    await _seed_user(pooled_urls["migrations"], tenant, "no-control-row@example.com")

    ctx = RequestContext(tenant_id=tenant, identity_id=uuid.uuid4())
    async with tenant_session(ctx) as session:
        assert session.get_bind() is get_engine().sync_engine
        emails = (await session.execute(text(_MEMBER_EMAILS))).scalars().all()
        assert emails == ["no-control-row@example.com"]
