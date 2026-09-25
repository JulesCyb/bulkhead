"""Real-Postgres integration test for the tenant-table registry (ADR-0010, Spec 9 / #65).

Verifies what a unit test on `TENANT_TABLES` alone cannot: that the migration actually applies
Row-Level Security to every table the registry lists (not a private per-migration list), and
that `unregistered_tenant_tables()` — the schema-introspection check the future erasure tool
depends on — really does introspect the live schema rather than trust the registry it is
checking. Pattern: `tests/test_rls_integration.py`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import ROLE_STATEMENT_TIMEOUT_MS
from app.db.tenant_tables import TENANT_TABLES, unregistered_tenant_tables

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


@pytest.fixture(scope="module")
def database_urls():
    """Same bootstrap as `tests/test_rls_integration.py`: `app_owner` runs the migrations."""
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
        "migrations": f"postgresql+asyncpg://app_owner@/postgres?host={sockdir}",
        "app": f"postgresql+asyncpg://app@/postgres?host={sockdir}",
        "superuser": f"postgresql+asyncpg://postgres@/postgres?host={sockdir}",
    }
    env = {**os.environ, "DATABASE_URL_MIGRATIONS": urls["migrations"], "DATABASE_URL": urls["app"]}
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"], check=True, env=env, timeout=120
    )
    yield urls
    server.cleanup()


async def test_every_registered_table_has_rls_enabled_and_forced(database_urls):
    """The migration iterates TENANT_TABLES, not a private list — every table it lists really
    has RLS enabled and forced, exactly as `migrations/versions/0001_initial.py` requires."""
    engine = create_async_engine(database_urls["migrations"])
    try:
        async with engine.connect() as conn:
            for table in TENANT_TABLES:
                relrowsecurity, relforcerowsecurity = (
                    await conn.execute(
                        text(
                            "SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
                            "WHERE oid = CAST(:table AS regclass)"
                        ),
                        {"table": table},
                    )
                ).one()
                assert relrowsecurity is True, f"{table} does not have RLS enabled"
                assert relforcerowsecurity is True, f"{table} does not have RLS forced"
    finally:
        await engine.dispose()


async def test_registry_has_full_coverage_of_the_public_schema(database_urls):
    """No tenant_id-bearing table in `public` currently escapes the registry."""
    engine = create_async_engine(database_urls["migrations"])
    try:
        async with engine.connect() as conn:
            assert await unregistered_tenant_tables(conn) == []
    finally:
        await engine.dispose()


async def test_unregistered_tenant_table_fails_the_check(database_urls):
    """A fixture table added only here, outside the registry, is caught immediately — the
    coverage gap the whole lifecycle tool depends on closes before anything is built on it."""
    engine = create_async_engine(database_urls["migrations"])
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    CREATE TABLE rogue_tenant_table (
                        id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                        tenant_id  uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE
                    )
                    """
                )
            )
        async with engine.connect() as conn:
            missing = await unregistered_tenant_tables(conn)
        assert missing == ["rogue_tenant_table"]
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DROP TABLE IF EXISTS rogue_tenant_table"))
        await engine.dispose()


async def test_control_plane_schema_is_excluded_by_construction(database_urls):
    """`control.tenants` (0002_control_plane_schema.py) carries tenant_id keyed straight to
    public.tenants, but the control-plane schema is operator-only and out of scope: the check
    only ever introspects `public`, so control.tenants is not required to register here."""
    engine = create_async_engine(database_urls["migrations"])
    try:
        async with engine.connect() as conn:
            has_control_tenants = (
                await conn.execute(
                    text(
                        "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                        "WHERE table_schema = 'control' AND table_name = 'tenants')"
                    )
                )
            ).scalar_one()
            assert has_control_tenants is True, "fixture assumption: control.tenants must exist"

            missing = await unregistered_tenant_tables(conn)
        assert "tenants" not in missing  # neither public.tenants nor control.tenants surface
        assert missing == []
    finally:
        await engine.dispose()
