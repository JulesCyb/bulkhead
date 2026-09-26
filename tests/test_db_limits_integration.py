"""Real Postgres test (`pgserver`, via `uv sync --group dbtest`) for connection-level bounds
(Spec 7 / #55): a runaway query is cut off at the database inside the tenant transaction, and
the app role carries its own statement-timeout and connection-limit independent of that.
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

from app.db.guard import ROLE_CONNECTION_LIMIT, ROLE_STATEMENT_TIMEOUT_MS

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
    pgdata = tempfile.mkdtemp(prefix="pgdata-")
    server = pgserver.get_server(pgdata, cleanup_mode="delete")
    sockdir = parse_qs(urlparse(server.get_uri()).query)["host"][0]
    # Mirrors docker/postgres/01-init.sh: role-level statement_timeout and CONNECTION LIMIT are
    # deployment settings, not something pgserver's default bootstrap sets up for us.
    _psql(
        server,
        "CREATE ROLE app LOGIN NOSUPERUSER NOBYPASSRLS "
        f"CONNECTION LIMIT {ROLE_CONNECTION_LIMIT}; "
        f"ALTER ROLE app SET statement_timeout = '{ROLE_STATEMENT_TIMEOUT_MS}ms'; "
        "GRANT USAGE ON SCHEMA public TO app; "
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE "
        "ON TABLES TO app;",
    )
    urls = {
        "migrations": f"postgresql+asyncpg://postgres@/postgres?host={sockdir}",
        "app": f"postgresql+asyncpg://app@/postgres?host={sockdir}",
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
    # A short per-transaction statement timeout so a `pg_sleep` engineered to outlast it proves
    # the cutoff happens inside the transaction, at the database, not in the application.
    monkeypatch.setenv("DB_STATEMENT_TIMEOUT_MS", "200")
    config.get_settings.cache_clear()
    db_session._engine = None
    db_session._session_factory = None
    yield
    config.get_settings.cache_clear()
    db_session._engine = None
    db_session._session_factory = None


async def test_runaway_query_is_cut_off_inside_the_tenant_transaction(app_settings, database_urls):
    from sqlalchemy.exc import DBAPIError

    from app.context import RequestContext
    from app.db.session import tenant_session

    engine = create_async_engine(database_urls["migrations"])
    tenant_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, 'A')"), {"id": tenant_id}
        )
    await engine.dispose()

    ctx = RequestContext(tenant_id=tenant_id, identity_id=uuid.uuid4())
    with pytest.raises(DBAPIError):
        async with tenant_session(ctx) as session:
            # 200ms statement_timeout above; this sleeps far longer, so it must raise.
            await session.execute(text("SELECT pg_sleep(2)"))


async def test_normal_query_completes_unaffected(app_settings, database_urls):
    from app.context import RequestContext
    from app.db.session import tenant_session

    engine = create_async_engine(database_urls["migrations"])
    tenant_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, 'A')"), {"id": tenant_id}
        )
    await engine.dispose()

    ctx = RequestContext(tenant_id=tenant_id, identity_id=uuid.uuid4())
    async with tenant_session(ctx) as session:
        result = await session.execute(text("SELECT 1"))
        assert result.scalar_one() == 1


async def test_app_role_has_its_own_statement_timeout_and_connection_limit(
    app_settings, database_urls
):
    """Role-level settings are independent of the per-transaction one set above (200ms): the
    role fact must reflect the deployment default (ROLE_STATEMENT_TIMEOUT_MS), not the
    request-scoped override."""
    engine = create_async_engine(database_urls["migrations"])
    async with engine.connect() as conn:
        rolconnlimit = (
            await conn.execute(text("SELECT rolconnlimit FROM pg_roles WHERE rolname = 'app'"))
        ).scalar_one()
        rolconfig = (
            await conn.execute(text("SELECT rolconfig FROM pg_roles WHERE rolname = 'app'"))
        ).scalar_one()
    await engine.dispose()

    assert rolconnlimit == ROLE_CONNECTION_LIMIT
    assert rolconfig is not None
    assert f"statement_timeout={ROLE_STATEMENT_TIMEOUT_MS}ms" in rolconfig
