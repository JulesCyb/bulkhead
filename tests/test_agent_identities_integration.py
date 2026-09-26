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

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine

from app.db.session import tenant_session  # noqa: E402
from app.repositories.agent_identities import AgentIdentityRepository  # noqa: E402
from app.repositories.memberships import MembershipRepository  # noqa: E402

pgserver = pytest.importorskip("pgserver")

from tests.support import cluster, environment, seed_tenant  # noqa: E402

_ = (cluster, environment)


async def test_creating_an_agent_identity_inserts_identity_of_kind_agent_and_its_membership(
    environment,
):
    """AC1: the identity row is kind 'agent' and its membership carries role 'agent' in exactly
    the calling tenant."""
    seeded = await seed_tenant(environment, roles=["admin"])
    ctx = seeded.ctx("admin")

    async with tenant_session(ctx) as session:
        created = await AgentIdentityRepository().create(session, ctx, name="nightly sync")
        memberships = await MembershipRepository().list_for_tenant(session, ctx)

    matching = [m for m in memberships if m.identity_id == created.identity_id]
    assert len(matching) == 1
    assert matching[0].role == "agent"
    assert matching[0].id == created.membership_id

    engine = create_async_engine(environment.superuser_url)
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


async def test_agent_identity_created_under_one_tenant_is_invisible_under_another(environment):
    """AC5: an agent identity's membership created under tenant A never appears in tenant B's
    membership listing, even though `control.identities` itself is global."""
    seeded_a = await seed_tenant(environment, roles=["admin"])
    seeded_b = await seed_tenant(environment, roles=["admin"])

    ctx_a = seeded_a.ctx("admin")
    async with tenant_session(ctx_a) as session:
        created = await AgentIdentityRepository().create(session, ctx_a, name="nightly sync")

    ctx_b = seeded_b.ctx("admin")
    async with tenant_session(ctx_b) as session:
        memberships_b = await MembershipRepository().list_for_tenant(session, ctx_b)
    assert all(m.identity_id != created.identity_id for m in memberships_b)

    # Still visible, unchanged, under its own tenant.
    async with tenant_session(ctx_a) as session:
        memberships_a = await MembershipRepository().list_for_tenant(session, ctx_a)
    assert any(m.identity_id == created.identity_id for m in memberships_a)


async def test_app_role_still_has_no_direct_insert_grant_on_control_identities(environment):
    """The narrow SECURITY DEFINER function is the only new capability this migration grants to
    `app` -- a direct INSERT against control.identities, bypassing the function entirely, must
    still fail exactly as it did before this ticket (ADR-0003 / #22)."""
    seeded = await seed_tenant(environment, roles=["admin"])
    ctx = seeded.ctx("admin")

    with pytest.raises(DBAPIError):
        async with tenant_session(ctx) as session:
            await session.execute(
                text(
                    "INSERT INTO control.identities (issuer, subject, kind) "
                    "VALUES ('forged', 'forged-subject', 'agent')"
                )
            )


async def test_create_agent_identity_requires_a_tenant_context(environment):
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
