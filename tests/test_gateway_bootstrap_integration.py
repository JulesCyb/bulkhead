"""Real bootstrap-script run against PostgreSQL (`pgserver`, via `uv sync --group dbtest`).

ADR-0009 / Spec 7 (#51): the model gateway (LiteLLM) gets its own role and its own database,
created by docker/postgres/01-init.sh, that carry no privilege on the application's tables. This
test runs the actual script (not a hand-copied mirror of its SQL, unlike test_rls_integration.py)
against an embedded Postgres instance and asserts the isolation from role/grant catalog facts.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import asyncpg
import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.ext.asyncio import create_async_engine

pgserver = pytest.importorskip("pgserver")

SCRIPT = Path(__file__).resolve().parent.parent / "docker" / "postgres" / "01-init.sh"


@pytest.fixture(scope="module")
def bootstrapped():
    """Runs the real docker/postgres/01-init.sh against an embedded Postgres instance, exactly
    as the Postgres container's entrypoint would (same env vars, same script), then hands back
    connection URLs for the roles it created.
    """
    from pgserver.postgres_server import POSTGRES_BIN_PATH

    pgdata = tempfile.mkdtemp(prefix="pgdata-")
    server = pgserver.get_server(pgdata, cleanup_mode="delete")
    sockdir = parse_qs(urlparse(server.get_uri()).query)["host"][0]

    env = {
        **os.environ,
        "PGHOST": sockdir,
        "POSTGRES_USER": "postgres",
        "POSTGRES_DB": "postgres",
        "APP_OWNER_DB_PASSWORD": "app_owner",
        "APP_DB_PASSWORD": "app",
        "GATEWAY_DB_PASSWORD": "gateway",
        "PATH": f"{POSTGRES_BIN_PATH}:{os.environ.get('PATH', '')}",
    }
    subprocess.run(["bash", str(SCRIPT)], check=True, env=env, capture_output=True, timeout=60)

    urls = {
        "superuser": f"postgresql+asyncpg://postgres@/postgres?host={sockdir}",
        "gateway": f"postgresql+asyncpg://gateway:gateway@/gateway?host={sockdir}",
        # Attempting to reach the application database as the gateway role.
        "app_as_gateway": f"postgresql+asyncpg://gateway:gateway@/postgres?host={sockdir}",
    }
    yield urls
    server.cleanup()


async def test_gateway_role_is_not_superuser_or_bypassrls(bootstrapped):
    engine = create_async_engine(bootstrapped["superuser"])
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT rolsuper, rolbypassrls, rolcreatedb, rolcreaterole FROM pg_roles "
                    "WHERE rolname = 'gateway'"
                )
            )
        ).one()
    await engine.dispose()
    assert row.rolsuper is False
    assert row.rolbypassrls is False
    assert row.rolcreatedb is False
    assert row.rolcreaterole is False


async def test_gateway_database_is_its_own_and_owned_by_the_gateway_role(bootstrapped):
    engine = create_async_engine(bootstrapped["superuser"])
    async with engine.connect() as conn:
        owner = (
            await conn.execute(
                text(
                    "SELECT r.rolname FROM pg_database d "
                    "JOIN pg_roles r ON r.oid = d.datdba WHERE d.datname = 'gateway'"
                )
            )
        ).scalar_one()
    await engine.dispose()
    assert owner == "gateway"


async def test_gateway_role_has_no_privilege_on_the_application_database_tables(bootstrapped):
    """Catalog fact, connected as the superuser: nothing in `information_schema` grants the
    gateway role a privilege on any object in the application's own database."""
    engine = create_async_engine(bootstrapped["superuser"])
    async with engine.connect() as conn:
        table_grants = (
            await conn.execute(
                text(
                    "SELECT table_name, privilege_type FROM information_schema.table_privileges "
                    "WHERE grantee = 'gateway'"
                )
            )
        ).all()
    await engine.dispose()
    assert table_grants == []


async def test_gateway_role_cannot_even_connect_to_the_application_database(bootstrapped):
    """The default PUBLIC CONNECT grant on the application database is revoked; only app_owner
    and app keep an explicit grant. The gateway role has no path into it at all."""
    engine = create_async_engine(bootstrapped["app_as_gateway"])
    # asyncpg raises this directly on connect, before SQLAlchemy's dialect wraps it.
    with pytest.raises(
        (OperationalError, DBAPIError, asyncpg.exceptions.InsufficientPrivilegeError)
    ):
        async with engine.connect():
            pass
    await engine.dispose()


async def test_gateway_role_can_connect_to_its_own_database(bootstrapped):
    engine = create_async_engine(bootstrapped["gateway"])
    async with engine.connect() as conn:
        current_db = (await conn.execute(text("SELECT current_database()"))).scalar_one()
    await engine.dispose()
    assert current_db == "gateway"
