"""Real isolation test for agent credentials (ADR-0005, Spec 6 / #45) against PostgreSQL +
pgvector (`pgserver`, via `uv sync --group dbtest`). Pattern: `tests/test_rls_integration.py`.

Proves what a unit test on `AgentCredentialRepository` alone cannot: that the RLS policy and
grants from migration 0021 actually isolate one tenant's credentials from another's at the
database layer, that the stored row never carries the plaintext secret, and that
`verify_and_touch` really does gate on both the hash and the revoked state while advancing
`last_used_at` only on success.
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
from app.context import RequestContext  # noqa: E402
from app.db.session import tenant_session  # noqa: E402
from app.repositories.agent_credentials import AgentCredentialRepository  # noqa: E402

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
    """Mirrors `tests/test_rls_integration.py`'s own fixture: app_owner/app roles, migrated to
    head with the real Alembic chain (including #45's 0021_agent_credentials)."""
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


async def _seed_tenant_admin_and_agent(url: str) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    """A tenant, an admin identity (the creator/actor), and an agent identity (what the
    credential belongs to) -- as the superuser, the same way test_rls_integration.py seeds
    fixtures that need real control.identities rows in place before a tenant_session() write."""
    engine = create_async_engine(url)
    tenant_id, admin_id, agent_identity_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, 'Acme')"), {"id": tenant_id}
        )
        for identity_id in (admin_id, agent_identity_id):
            await conn.execute(
                text(
                    "INSERT INTO control.identities (id, issuer, subject) "
                    "VALUES (:id, 'seed', :sub)"
                ),
                {"id": identity_id, "sub": str(identity_id)},
            )
    await engine.dispose()
    return tenant_id, admin_id, agent_identity_id


async def test_credential_created_under_one_tenant_is_invisible_under_another(
    app_settings, database_urls
):
    """AC1: a credential created under tenant A cannot be listed, verified, or revoked under
    tenant B's session, even by its exact identifier."""
    tenant_a, admin_a, agent_a = await _seed_tenant_admin_and_agent(database_urls["superuser"])
    tenant_b, admin_b, _ = await _seed_tenant_admin_and_agent(database_urls["superuser"])

    ctx_a = RequestContext(tenant_id=tenant_a, identity_id=admin_a)
    repo = AgentCredentialRepository()
    async with tenant_session(ctx_a) as session:
        issued = await repo.create(session, ctx_a, identity_id=agent_a, name="nightly sync")

    ctx_b = RequestContext(tenant_id=tenant_b, identity_id=admin_b)
    async with tenant_session(ctx_b) as session:
        listing = await repo.list_for_tenant(session, ctx_b)
        assert listing == []

        verified = await repo.verify_and_touch(
            session, ctx_b, public_id=issued.public_id, secret=issued.secret
        )
        assert verified is None

        revoked = await repo.revoke(session, ctx_b, credential_id=issued.id)
        assert revoked is False

    # And it is still perfectly usable back under its own tenant.
    async with tenant_session(ctx_a) as session:
        verified = await repo.verify_and_touch(
            session, ctx_a, public_id=issued.public_id, secret=issued.secret
        )
        assert verified is not None
        assert verified.identity_id == agent_a


async def test_stored_row_never_contains_the_plaintext_secret(app_settings, database_urls):
    """AC2: the row holds only the hash plus the public identifier -- inspected directly, at
    the database layer, not through the repository that produced it."""
    tenant_id, admin_id, agent_id = await _seed_tenant_admin_and_agent(database_urls["superuser"])
    ctx = RequestContext(tenant_id=tenant_id, identity_id=admin_id)
    repo = AgentCredentialRepository()
    async with tenant_session(ctx) as session:
        issued = await repo.create(session, ctx, identity_id=agent_id, name="helpdesk bot")

    engine = create_async_engine(database_urls["superuser"])
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text("SELECT public_id, secret_hash FROM agent_credentials WHERE id = :id"),
                {"id": issued.id},
            )
        ).one()
    await engine.dispose()

    assert row.public_id == issued.public_id
    assert row.secret_hash != issued.secret
    assert issued.secret not in row.secret_hash
    assert len(row.secret_hash) == 64  # hex-encoded SHA-256 digest


async def test_revoked_credential_fails_verification_from_that_moment_on(
    app_settings, database_urls
):
    """AC3: verification succeeds before revocation and fails after, for the same secret."""
    tenant_id, admin_id, agent_id = await _seed_tenant_admin_and_agent(database_urls["superuser"])
    ctx = RequestContext(tenant_id=tenant_id, identity_id=admin_id)
    repo = AgentCredentialRepository()

    async with tenant_session(ctx) as session:
        issued = await repo.create(session, ctx, identity_id=agent_id, name="reporting job")
        verified_before = await repo.verify_and_touch(
            session, ctx, public_id=issued.public_id, secret=issued.secret
        )
        assert verified_before is not None

        revoked = await repo.revoke(session, ctx, credential_id=issued.id)
        assert revoked is True

        verified_after = await repo.verify_and_touch(
            session, ctx, public_id=issued.public_id, secret=issued.secret
        )
        assert verified_after is None


async def test_last_used_advances_only_after_a_successful_verification(app_settings, database_urls):
    """AC4: a failed attempt (wrong secret) leaves last_used_at untouched; a subsequent
    successful attempt advances it."""
    tenant_id, admin_id, agent_id = await _seed_tenant_admin_and_agent(database_urls["superuser"])
    ctx = RequestContext(tenant_id=tenant_id, identity_id=admin_id)
    repo = AgentCredentialRepository()

    async with tenant_session(ctx) as session:
        issued = await repo.create(session, ctx, identity_id=agent_id, name="ci runner")

    async def _last_used_at() -> object:
        engine = create_async_engine(database_urls["superuser"])
        async with engine.connect() as conn:
            value = (
                await conn.execute(
                    text("SELECT last_used_at FROM agent_credentials WHERE id = :id"),
                    {"id": issued.id},
                )
            ).scalar_one()
        await engine.dispose()
        return value

    assert await _last_used_at() is None

    async with tenant_session(ctx) as session:
        failed = await repo.verify_and_touch(
            session, ctx, public_id=issued.public_id, secret="not-the-right-secret"
        )
        assert failed is None
    assert await _last_used_at() is None

    async with tenant_session(ctx) as session:
        succeeded = await repo.verify_and_touch(
            session, ctx, public_id=issued.public_id, secret=issued.secret
        )
        assert succeeded is not None
    first_touch = await _last_used_at()
    assert first_touch is not None


async def test_revoking_leaves_identity_and_sibling_credentials_untouched(
    app_settings, database_urls
):
    """AC5: revoking one credential never touches the agent identity's membership row (out of
    scope here, not created by this repository) nor any other credential issued to the same
    agent identity."""
    tenant_id, admin_id, agent_id = await _seed_tenant_admin_and_agent(database_urls["superuser"])
    ctx = RequestContext(tenant_id=tenant_id, identity_id=admin_id)
    repo = AgentCredentialRepository()

    async with tenant_session(ctx) as session:
        first = await repo.create(session, ctx, identity_id=agent_id, name="first")
        second = await repo.create(session, ctx, identity_id=agent_id, name="second")

        revoked = await repo.revoke(session, ctx, credential_id=first.id)
        assert revoked is True

        listing = {record.id: record for record in await repo.list_for_tenant(session, ctx)}
    assert listing[first.id].revoked_at is not None
    assert listing[second.id].revoked_at is None
    assert listing[first.id].identity_id == agent_id
    assert listing[second.id].identity_id == agent_id

    # The identity itself is still a perfectly ordinary control.identities row.
    engine = create_async_engine(database_urls["superuser"])
    async with engine.connect() as conn:
        still_there = (
            await conn.execute(
                text("SELECT count(*) FROM control.identities WHERE id = :id"), {"id": agent_id}
            )
        ).scalar_one()
    await engine.dispose()
    assert still_there == 1


async def test_table_has_forced_rls_with_tenant_policy_and_no_delete_grant(
    app_settings, database_urls
):
    """AC6: the same tenant-scoping, forced RLS, and grant shape as every other tenant table
    (CLAUDE.md rule 2) -- verified at the database layer, plus a cross-tenant write attempt
    directly against the table (bypassing the repository) is rejected by the policy itself."""
    tenant_a, admin_a, agent_a = await _seed_tenant_admin_and_agent(database_urls["superuser"])
    tenant_b, _, _ = await _seed_tenant_admin_and_agent(database_urls["superuser"])

    engine = create_async_engine(database_urls["migrations"])
    async with engine.connect() as conn:
        relrowsecurity, relforcerowsecurity = (
            await conn.execute(
                text(
                    "SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
                    "WHERE oid = CAST('agent_credentials' AS regclass)"
                )
            )
        ).one()
        assert relrowsecurity is True
        assert relforcerowsecurity is True

        grants = {
            row.privilege_type
            for row in (
                await conn.execute(
                    text(
                        "SELECT privilege_type FROM information_schema.role_table_grants "
                        "WHERE table_schema = 'public' AND table_name = 'agent_credentials' "
                        "AND grantee = 'app'"
                    )
                )
            )
        }
    await engine.dispose()
    assert grants == {"SELECT", "INSERT", "UPDATE"}
    assert "DELETE" not in grants

    ctx_a = RequestContext(tenant_id=tenant_a, identity_id=admin_a)
    repo = AgentCredentialRepository()
    async with tenant_session(ctx_a) as session:
        issued = await repo.create(session, ctx_a, identity_id=agent_a, name="direct-access probe")

    # Directly against the table as the app role, but under tenant B's session context: the
    # policy's USING clause must block the read regardless of the repository's own WHERE clause.
    ctx_b = RequestContext(tenant_id=tenant_b, identity_id=admin_a)
    async with tenant_session(ctx_b) as session:
        row = (
            await session.execute(
                text("SELECT id FROM agent_credentials WHERE id = :id"), {"id": issued.id}
            )
        ).first()
    assert row is None
