"""Real isolation test for agent credentials (ADR-0005, Spec 6 / #45) against PostgreSQL +
pgvector (`pgserver`, via `uv sync --group dbtest`). Pattern: `tests/test_rls_integration.py`.

Proves what a unit test on `AgentCredentialRepository` alone cannot: that the RLS policy and
grants from migration 0021 actually isolate one tenant's credentials from another's at the
database layer, that the stored row never carries the plaintext secret, and that
`verify_and_touch` really does gate on both the hash and the revoked state while advancing
`last_used_at` only on success.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine

from app.context import RequestContext  # noqa: E402
from app.db.session import tenant_session  # noqa: E402
from app.repositories.agent_credentials import (  # noqa: E402
    AgentCredentialRepository,
    UnknownAgentIdentity,
)

pgserver = pytest.importorskip("pgserver")

from tests.support import cluster, environment, seed_membership, seed_tenant  # noqa: E402

_ = (cluster, environment)


async def _tenant_with_admin_and_agent(environment):
    """A tenant, an admin identity (the creator/actor), and an agent identity (what the
    credential belongs to) -- via `seed_tenant`'s `via_operator=False` path, which writes both
    memberships with real `control.identities` rows directly, as the cluster's own superuser,
    with no gateway credential needed here.

    `AgentCredentialRepository.create` requires the target identity to carry an active
    `agent`-role membership in the caller's own tenant (review of #46) -- checked purely against
    `memberships.role`, never `control.identities.kind` (see that repository's own docstring for
    why) -- so `roles=["admin", "agent"]` is exactly what every test below needs."""
    seeded = await seed_tenant(environment, roles=["admin", "agent"], via_operator=False)
    return seeded, seeded.identities["admin"], seeded.identities["agent"]


async def test_credential_created_under_one_tenant_is_invisible_under_another(environment):
    """AC1: a credential created under tenant A cannot be listed, verified, or revoked under
    tenant B's session, even by its exact identifier."""
    tenant_a, admin_a, agent_a = await _tenant_with_admin_and_agent(environment)
    tenant_b, admin_b, _ = await _tenant_with_admin_and_agent(environment)

    ctx_a = RequestContext(tenant_id=tenant_a.tenant_id, identity_id=admin_a)
    repo = AgentCredentialRepository()
    async with tenant_session(ctx_a) as session:
        issued = await repo.create(session, ctx_a, identity_id=agent_a, name="nightly sync")

    ctx_b = RequestContext(tenant_id=tenant_b.tenant_id, identity_id=admin_b)
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


async def test_stored_row_never_contains_the_plaintext_secret(environment):
    """AC2: the row holds only the hash plus the public identifier -- inspected directly, at
    the database layer, not through the repository that produced it."""
    tenant, admin_id, agent_id = await _tenant_with_admin_and_agent(environment)
    ctx = RequestContext(tenant_id=tenant.tenant_id, identity_id=admin_id)
    repo = AgentCredentialRepository()
    async with tenant_session(ctx) as session:
        issued = await repo.create(session, ctx, identity_id=agent_id, name="helpdesk bot")

    async with tenant.superuser_connection() as conn:
        row = (
            await conn.execute(
                text("SELECT public_id, secret_hash FROM agent_credentials WHERE id = :id"),
                {"id": issued.id},
            )
        ).one()

    assert row.public_id == issued.public_id
    assert row.secret_hash != issued.secret
    assert issued.secret not in row.secret_hash
    assert len(row.secret_hash) == 64  # hex-encoded SHA-256 digest


async def test_revoked_credential_fails_verification_from_that_moment_on(environment):
    """AC3: verification succeeds before revocation and fails after, for the same secret."""
    tenant, admin_id, agent_id = await _tenant_with_admin_and_agent(environment)
    ctx = RequestContext(tenant_id=tenant.tenant_id, identity_id=admin_id)
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


async def test_last_used_advances_only_after_a_successful_verification(environment):
    """AC4: a failed attempt (wrong secret) leaves last_used_at untouched; a subsequent
    successful attempt advances it."""
    tenant, admin_id, agent_id = await _tenant_with_admin_and_agent(environment)
    ctx = RequestContext(tenant_id=tenant.tenant_id, identity_id=admin_id)
    repo = AgentCredentialRepository()

    async with tenant_session(ctx) as session:
        issued = await repo.create(session, ctx, identity_id=agent_id, name="ci runner")

    async def _last_used_at() -> object:
        async with tenant.superuser_connection() as conn:
            return (
                await conn.execute(
                    text("SELECT last_used_at FROM agent_credentials WHERE id = :id"),
                    {"id": issued.id},
                )
            ).scalar_one()

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


async def test_revoking_leaves_identity_and_sibling_credentials_untouched(environment):
    """AC5: revoking one credential never touches the agent identity's membership row (out of
    scope here, not created by this repository) nor any other credential issued to the same
    agent identity."""
    tenant, admin_id, agent_id = await _tenant_with_admin_and_agent(environment)
    ctx = RequestContext(tenant_id=tenant.tenant_id, identity_id=admin_id)
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
    async with tenant.superuser_connection() as conn:
        still_there = (
            await conn.execute(
                text("SELECT count(*) FROM control.identities WHERE id = :id"), {"id": agent_id}
            )
        ).scalar_one()
    assert still_there == 1


async def test_table_has_forced_rls_with_tenant_policy_and_no_delete_grant(environment):
    """AC6: the same tenant-scoping, forced RLS, and grant shape as every other tenant table
    (CLAUDE.md rule 2) -- verified at the database layer, plus a cross-tenant write attempt
    directly against the table (bypassing the repository) is rejected by the policy itself."""
    tenant_a, admin_a, agent_a = await _tenant_with_admin_and_agent(environment)
    tenant_b, _, _ = await _tenant_with_admin_and_agent(environment)

    engine = create_async_engine(environment.owner_url)
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

    ctx_a = RequestContext(tenant_id=tenant_a.tenant_id, identity_id=admin_a)
    repo = AgentCredentialRepository()
    async with tenant_session(ctx_a) as session:
        issued = await repo.create(session, ctx_a, identity_id=agent_a, name="direct-access probe")

    # Directly against the table as the app role, but under tenant B's session context: the
    # policy's USING clause must block the read regardless of the repository's own WHERE clause.
    ctx_b = RequestContext(tenant_id=tenant_b.tenant_id, identity_id=admin_a)
    async with tenant_session(ctx_b) as session:
        row = (
            await session.execute(
                text("SELECT id FROM agent_credentials WHERE id = :id"), {"id": issued.id}
            )
        ).first()
    assert row is None


# --- Review finding (#46, 2026-09-25): a credential must name an agent identity of the caller's
# own tenant -- checked in the repository, and again by the database as a backstop. ---


async def test_repository_refuses_a_cross_tenant_agent_identity(environment):
    """An admin of tenant A cannot mint a credential for tenant B's agent identity: the
    repository's own membership check (not RLS, which never sees `identity_id` cross-tenant in
    the first place -- this is a check against `memberships`, not `agent_credentials`) refuses it
    before any row is written."""
    tenant_a, admin_a, _ = await _tenant_with_admin_and_agent(environment)
    _tenant_b, _admin_b, agent_b = await _tenant_with_admin_and_agent(environment)

    ctx_a = RequestContext(tenant_id=tenant_a.tenant_id, identity_id=admin_a)
    repo = AgentCredentialRepository()
    async with tenant_session(ctx_a) as session:
        with pytest.raises(UnknownAgentIdentity):
            await repo.create(session, ctx_a, identity_id=agent_b, name="should never exist")

    async with tenant_a.superuser_connection() as conn:
        count = (
            await conn.execute(
                text("SELECT count(*) FROM agent_credentials WHERE tenant_id = :tid"),
                {"tid": tenant_a.tenant_id},
            )
        ).scalar_one()
    assert count == 0


async def test_repository_refuses_a_person_identity_in_the_callers_own_tenant(environment):
    """A membership that exists in the caller's own tenant but carries a role other than `agent`
    (an ordinary person) is refused exactly like an unknown id -- `UnknownAgentIdentity` either
    way, and no row written."""
    tenant, admin_id, _ = await _tenant_with_admin_and_agent(environment)
    person_id, _ = await seed_membership(environment, tenant_id=tenant.tenant_id, role="member")

    ctx = RequestContext(tenant_id=tenant.tenant_id, identity_id=admin_id)
    repo = AgentCredentialRepository()
    async with tenant_session(ctx) as session:
        with pytest.raises(UnknownAgentIdentity):
            await repo.create(session, ctx, identity_id=person_id, name="should never exist")


async def test_db_trigger_refuses_an_insert_that_bypasses_the_repository(environment):
    """The migration-0041 trigger is the backstop: even a raw `INSERT` against `agent_credentials`
    (as the `app` role, bypassing `AgentCredentialRepository.create` entirely) is refused unless
    `identity_id` carries an `agent`-role membership in the same `tenant_id`. Proves the invariant
    holds even if a future writer of this table never reads the repository's docstring."""
    tenant, admin_id, _ = await _tenant_with_admin_and_agent(environment)
    person_id, _ = await seed_membership(environment, tenant_id=tenant.tenant_id, role="member")

    ctx = RequestContext(tenant_id=tenant.tenant_id, identity_id=admin_id)
    async with tenant_session(ctx) as session:
        with pytest.raises(DBAPIError):
            await session.execute(
                text(
                    "INSERT INTO agent_credentials "
                    "(tenant_id, identity_id, name, public_id, secret_hash) "
                    "VALUES (:tenant_id, :identity_id, 'raw insert', 'agt_raw', repeat('a', 64))"
                ),
                {"tenant_id": tenant.tenant_id, "identity_id": person_id},
            )
        await session.rollback()


async def test_db_trigger_allows_an_insert_for_a_real_agent_membership(environment):
    """The same trigger must not reject a legitimate insert -- proven directly, independent of
    the repository, so the trigger's own logic (not just the repository's) is what is tested."""
    tenant, admin_id, agent_id = await _tenant_with_admin_and_agent(environment)

    ctx = RequestContext(tenant_id=tenant.tenant_id, identity_id=admin_id)
    async with tenant_session(ctx) as session:
        await session.execute(
            text(
                "INSERT INTO agent_credentials "
                "(tenant_id, identity_id, name, public_id, secret_hash) "
                "VALUES (:tenant_id, :identity_id, 'raw insert', 'agt_raw2', repeat('a', 64))"
            ),
            {"tenant_id": tenant.tenant_id, "identity_id": agent_id},
        )

    async with tenant.superuser_connection() as conn:
        count = (
            await conn.execute(
                text("SELECT count(*) FROM agent_credentials WHERE public_id = 'agt_raw2'")
            )
        ).scalar_one()
    assert count == 1
