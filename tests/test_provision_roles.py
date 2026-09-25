"""Embedded-Postgres integration test for `scripts/provision_roles.py` (Spec 9 / #67): the
standalone alternative to `docker/postgres/01-init.sh` for managed Postgres (RDS, Neon, Supabase,
Cloud SQL) with no first-boot container hook. Verifies it creates the same `app_owner`/`app`
roles and grants the container's init script creates, and that running it twice is a no-op the
second time.
"""

from __future__ import annotations

import tempfile
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from scripts.provision_roles import provision

pgserver = pytest.importorskip("pgserver")


@pytest.fixture
def admin_url():
    pgdata = tempfile.mkdtemp(prefix="pgdata-provision-")
    server = pgserver.get_server(pgdata, cleanup_mode="delete")
    sockdir = parse_qs(urlparse(server.get_uri()).query)["host"][0]
    yield f"postgresql://postgres@/postgres?host={sockdir}"
    server.cleanup()


async def _role_row(admin_url: str, rolname: str):
    engine = create_async_engine(admin_url.replace("postgresql://", "postgresql+asyncpg://"))
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT rolsuper, rolbypassrls, rolcreatedb, rolcreaterole, rolconnlimit "
                    "FROM pg_roles WHERE rolname = :rolname"
                ),
                {"rolname": rolname},
            )
        ).one_or_none()
    await engine.dispose()
    return row


async def _grants_snapshot(admin_url: str):
    engine = create_async_engine(admin_url.replace("postgresql://", "postgresql+asyncpg://"))
    async with engine.connect() as conn:
        schema_owner = (
            await conn.execute(
                text(
                    "SELECT r.rolname FROM pg_namespace n "
                    "JOIN pg_roles r ON r.oid = n.nspowner WHERE n.nspname = 'public'"
                )
            )
        ).scalar_one()
        privileges = (
            await conn.execute(
                text(
                    "SELECT "
                    "has_database_privilege('app_owner', current_database(), 'CONNECT'), "
                    "has_database_privilege('app_owner', current_database(), 'CREATE'), "
                    "has_database_privilege('app', current_database(), 'CONNECT'), "
                    "has_schema_privilege('app', 'public', 'USAGE'), "
                    "has_schema_privilege('app', 'public', 'CREATE')"
                )
            )
        ).one()
        timeout = (
            await conn.execute(
                text(
                    "SELECT setting FROM pg_db_role_setting drs "
                    "JOIN pg_roles r ON r.oid = drs.setrole "
                    "CROSS JOIN LATERAL unnest(drs.setconfig) AS setting "
                    "WHERE r.rolname = 'app'"
                )
            )
        ).all()
    await engine.dispose()
    return schema_owner, tuple(privileges), timeout


async def test_creates_app_owner_and_app_roles_matching_init_script(admin_url):
    await provision(admin_url)

    app_owner = await _role_row(admin_url, "app_owner")
    assert app_owner is not None
    assert app_owner.rolsuper is False
    assert app_owner.rolbypassrls is False
    assert app_owner.rolcreatedb is False
    assert app_owner.rolcreaterole is False

    app = await _role_row(admin_url, "app")
    assert app is not None
    assert app.rolsuper is False
    assert app.rolbypassrls is False
    assert app.rolcreatedb is False
    assert app.rolcreaterole is False
    assert app.rolconnlimit == 50

    schema_owner, privileges, _ = await _grants_snapshot(admin_url)
    assert schema_owner == "app_owner"
    # CONNECT for both roles, CREATE on the database for app_owner only (the `control` schema).
    assert privileges[:3] == (True, True, True)
    assert privileges[4] is False  # app has no CREATE on schema public


async def test_app_statement_timeout_matches_role_bootstrap(admin_url):
    from app.config import ROLE_STATEMENT_TIMEOUT_MS

    await provision(admin_url)

    engine = create_async_engine(
        f"postgresql+asyncpg://app@/postgres?host={parse_qs(urlparse(admin_url).query)['host'][0]}"
    )
    async with engine.connect() as conn:
        timeout_ms = (
            await conn.execute(
                text("SELECT setting FROM pg_settings WHERE name = 'statement_timeout'")
            )
        ).scalar_one()
    await engine.dispose()
    assert int(timeout_ms) == ROLE_STATEMENT_TIMEOUT_MS


async def test_running_twice_is_a_noop(admin_url):
    await provision(admin_url)
    before = await _grants_snapshot(admin_url)
    before_app_owner = await _role_row(admin_url, "app_owner")
    before_app = await _role_row(admin_url, "app")

    await provision(admin_url)  # second run: must not error or change anything

    after = await _grants_snapshot(admin_url)
    after_app_owner = await _role_row(admin_url, "app_owner")
    after_app = await _role_row(admin_url, "app")

    assert before == after
    assert tuple(before_app_owner) == tuple(after_app_owner)
    assert tuple(before_app) == tuple(after_app)


async def test_app_role_can_connect_and_use_schema(admin_url):
    """The app role's own grants (CONNECT + USAGE on public), exercised as that role."""
    await provision(admin_url)
    sockdir = parse_qs(urlparse(admin_url).query)["host"][0]
    engine = create_async_engine(f"postgresql+asyncpg://app@/postgres?host={sockdir}")
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    await engine.dispose()
