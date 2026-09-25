"""Real isolation test for agent identity creation (ADR-0005, Spec 6 / #46) against PostgreSQL +
pgvector (`pgserver`, via `uv sync --group dbtest`). Pattern:
`tests/test_agent_credentials_integration.py`.

Proves what a unit test with faked repositories cannot: that `control.create_agent_identity`
(migration 0032) really does insert a `control.identities` row of kind `agent` plus its
`agent`-role `memberships` row in exactly the caller's own tenant, that `app` still has no direct
`INSERT` grant on `control.identities`, and that a membership created under one tenant's session
never shows up under another's.
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
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import ROLE_STATEMENT_TIMEOUT_MS  # noqa: E402
from app.context import RequestContext  # noqa: E402
from app.db.session import tenant_session  # noqa: E402
from app.repositories.agent_identities import AgentIdentityRepository  # noqa: E402
from app.repositories.memberships import MembershipRepository  # noqa: E402

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
    """Mirrors `tests/test_agent_credentials_integration.py`'s own fixture: app_owner/app roles,
    migrated to head with the real Alembic chain (including #46's 0032_agent_identity_creation)."""
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


async def _seed_tenant_and_admin(url: str) -> tuple[uuid.UUID, uuid.UUID]:
    """A tenant and an admin identity (the actor creating the agent identity) -- as the
    superuser, the same way test_agent_credentials_integration.py seeds fixtures that need real
    control.identities rows in place before a tenant_session() write."""
    engine = create_async_engine(url)
    tenant_id, admin_id = uuid.uuid4(), uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, 'Acme')"), {"id": tenant_id}
        )
        await conn.execute(
            text("INSERT INTO control.identities (id, issuer, subject) VALUES (:id, 'seed', :sub)"),
            {"id": admin_id, "sub": str(admin_id)},
        )
    await engine.dispose()
    return tenant_id, admin_id


async def test_creating_an_agent_identity_inserts_identity_of_kind_agent_and_its_membership(
    app_settings, database_urls
):
    """AC1: the identity row is kind 'agent' and its membership carries role 'agent' in exactly
    the calling tenant."""
    tenant_id, admin_id = await _seed_tenant_and_admin(database_urls["superuser"])
    ctx = RequestContext(tenant_id=tenant_id, identity_id=admin_id)

    async with tenant_session(ctx) as session:
        created = await AgentIdentityRepository().create(session, ctx, name="nightly sync")
        memberships = await MembershipRepository().list_for_tenant(session, ctx)

    matching = [m for m in memberships if m.identity_id == created.identity_id]
    assert len(matching) == 1
    assert matching[0].role == "agent"
    assert matching[0].id == created.membership_id

    engine = create_async_engine(database_urls["superuser"])
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text("SELECT kind, issuer, subject FROM control.identities WHERE id = :id"),
                {"id": created.identity_id},
            )
        ).one()
    await engine.dispose()
    assert row.kind == "agent"
    # Issuer/subject are synthesized by the function itself, never derived from caller input.
    assert row.issuer == "agent"
    assert row.subject


async def test_agent_identity_created_under_one_tenant_is_invisible_under_another(
    app_settings, database_urls
):
    """AC5: an agent identity's membership created under tenant A never appears in tenant B's
    membership listing, even though `control.identities` itself is global."""
    tenant_a, admin_a = await _seed_tenant_and_admin(database_urls["superuser"])
    tenant_b, admin_b = await _seed_tenant_and_admin(database_urls["superuser"])

    ctx_a = RequestContext(tenant_id=tenant_a, identity_id=admin_a)
    async with tenant_session(ctx_a) as session:
        created = await AgentIdentityRepository().create(session, ctx_a, name="nightly sync")

    ctx_b = RequestContext(tenant_id=tenant_b, identity_id=admin_b)
    async with tenant_session(ctx_b) as session:
        memberships_b = await MembershipRepository().list_for_tenant(session, ctx_b)
    assert all(m.identity_id != created.identity_id for m in memberships_b)

    # Still visible, unchanged, under its own tenant.
    async with tenant_session(ctx_a) as session:
        memberships_a = await MembershipRepository().list_for_tenant(session, ctx_a)
    assert any(m.identity_id == created.identity_id for m in memberships_a)


async def test_app_role_still_has_no_direct_insert_grant_on_control_identities(
    app_settings, database_urls
):
    """The narrow SECURITY DEFINER function is the only new capability this migration grants to
    `app` -- a direct INSERT against control.identities, bypassing the function entirely, must
    still fail exactly as it did before this ticket (ADR-0003 / #22)."""
    tenant_id, admin_id = await _seed_tenant_and_admin(database_urls["superuser"])
    ctx = RequestContext(tenant_id=tenant_id, identity_id=admin_id)

    with pytest.raises(DBAPIError):
        async with tenant_session(ctx) as session:
            await session.execute(
                text(
                    "INSERT INTO control.identities (issuer, subject, kind) "
                    "VALUES ('forged', 'forged-subject', 'agent')"
                )
            )


async def test_create_agent_identity_requires_a_tenant_context(app_settings, database_urls):
    """The function reads the tenant from `app.tenant_id`, set by `tenant_session`; called
    without one (a control-plane-style session), it must fail closed rather than default to some
    tenant, or none."""
    from app.db.session import control_session

    with pytest.raises(DBAPIError):
        async with control_session() as session:
            await session.execute(
                text("SELECT identity_id, membership_id FROM control.create_agent_identity(:name)"),
                {"name": "orphan"},
            )
