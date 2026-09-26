"""Real isolation test for standing grants (ADR-0005, ADR-0007, Spec 5 / #38) against PostgreSQL
+ pgvector (`pgserver`, via `uv sync --group dbtest`). Pattern:
`tests/test_agent_credentials_integration.py`.

Proves what a unit test on `StandingGrantRepository` alone cannot: that the target-role check and
the partial unique index really run against live Postgres state, that the RLS policy from
migration 0031 isolates one tenant's grants from another's, and that the admin-only role gate on
`app/tools/standing_grants.py` really refuses a non-admin caller end to end.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from app.context import RequestContext  # noqa: E402
from app.db.session import tenant_session  # noqa: E402
from app.repositories.standing_grants import (  # noqa: E402
    NotAnAgentMembership,
    StandingGrantRepository,
)
from app.tools.standing_grants import (  # noqa: E402
    create_standing_grant,
    list_standing_grants,
    revoke_standing_grant,
)

pgserver = pytest.importorskip("pgserver")

from tests.support import cluster, environment, seed_membership, seed_tenant  # noqa: E402

_ = (cluster, environment)


async def test_creating_a_grant_for_a_non_agent_membership_is_refused(environment):
    """AC1: a membership that does not carry the agent role cannot be granted a standing
    permission."""
    seeded = await seed_tenant(environment)
    _, admin_membership = await seed_membership(
        environment, tenant_id=seeded.tenant_id, role="admin"
    )
    _, member_membership = await seed_membership(
        environment, tenant_id=seeded.tenant_id, role="member"
    )

    ctx = RequestContext(
        tenant_id=seeded.tenant_id, identity_id=uuid.uuid4(), roles=frozenset({"admin"})
    )
    repo = StandingGrantRepository()
    with pytest.raises(NotAnAgentMembership):
        async with tenant_session(ctx) as session:
            await repo.create(
                session,
                ctx,
                agent_membership_id=member_membership,
                tool_name="send_invoice",
                granted_by=admin_membership,
            )


async def test_creating_a_grant_on_behalf_of_a_non_admin_is_refused(environment):
    """AC2: the same role check Spec 3 (#26) provides refuses a non-admin caller, before any
    data access."""
    seeded = await seed_tenant(environment)
    _, admin_membership = await seed_membership(
        environment, tenant_id=seeded.tenant_id, role="admin"
    )
    _, agent_membership = await seed_membership(
        environment, tenant_id=seeded.tenant_id, role="agent"
    )

    for role in ("member", "support", "agent"):
        ctx = RequestContext(
            tenant_id=seeded.tenant_id, identity_id=uuid.uuid4(), roles=frozenset({role})
        )
        with pytest.raises(PermissionError):
            await create_standing_grant(
                ctx,
                agent_membership_id=agent_membership,
                tool_name="send_invoice",
                granted_by=admin_membership,
            )


async def test_second_active_grant_for_same_tool_is_rejected_a_different_tool_succeeds(
    environment,
):
    """AC3: a partial uniqueness rule rejects a second active grant for the same tenant, agent
    identity, and tool, while a second grant naming a different tool for the same agent identity
    succeeds."""
    seeded = await seed_tenant(environment)
    _, admin_membership = await seed_membership(
        environment, tenant_id=seeded.tenant_id, role="admin"
    )
    _, agent_membership = await seed_membership(
        environment, tenant_id=seeded.tenant_id, role="agent"
    )
    ctx = RequestContext(
        tenant_id=seeded.tenant_id, identity_id=uuid.uuid4(), roles=frozenset({"admin"})
    )
    repo = StandingGrantRepository()

    async with tenant_session(ctx) as session:
        await repo.create(
            session,
            ctx,
            agent_membership_id=agent_membership,
            tool_name="send_invoice",
            granted_by=admin_membership,
        )

    with pytest.raises(IntegrityError):
        async with tenant_session(ctx) as session:
            await repo.create(
                session,
                ctx,
                agent_membership_id=agent_membership,
                tool_name="send_invoice",
                granted_by=admin_membership,
            )

    async with tenant_session(ctx) as session:
        second = await repo.create(
            session,
            ctx,
            agent_membership_id=agent_membership,
            tool_name="delete_document",
            granted_by=admin_membership,
        )
        assert second.tool_name == "delete_document"


async def test_revoking_records_who_and_when_never_deletes_and_lookup_returns_none(environment):
    """AC4: revoking a grant records the revoking membership and a timestamp on the same row
    (never deletes it), and the very next active-grant lookup for that agent identity and tool
    returns none."""
    seeded = await seed_tenant(environment)
    _, admin_membership = await seed_membership(
        environment, tenant_id=seeded.tenant_id, role="admin"
    )
    _, revoking_admin_membership = await seed_membership(
        environment, tenant_id=seeded.tenant_id, role="admin"
    )
    _, agent_membership = await seed_membership(
        environment, tenant_id=seeded.tenant_id, role="agent"
    )
    ctx = RequestContext(
        tenant_id=seeded.tenant_id, identity_id=uuid.uuid4(), roles=frozenset({"admin"})
    )
    repo = StandingGrantRepository()

    async with tenant_session(ctx) as session:
        grant = await repo.create(
            session,
            ctx,
            agent_membership_id=agent_membership,
            tool_name="send_invoice",
            granted_by=admin_membership,
        )
        grant_id = grant.id

    async with tenant_session(ctx) as session:
        active_before = await repo.get_active(
            session, ctx, agent_membership_id=agent_membership, tool_name="send_invoice"
        )
        assert active_before is not None

        revoked = await repo.revoke(
            session, ctx, grant_id=grant_id, revoked_by=revoking_admin_membership
        )
        assert revoked is True

        active_after = await repo.get_active(
            session, ctx, agent_membership_id=agent_membership, tool_name="send_invoice"
        )
        assert active_after is None

    # Never deleted -- still present, directly at the database layer, with who/when recorded.
    engine = create_async_engine(environment.superuser_url)
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text("SELECT revoked_at, revoked_by FROM standing_grants WHERE id = :id"),
                {"id": grant_id},
            )
        ).one()
    await engine.dispose()
    assert row.revoked_at is not None
    assert row.revoked_by == revoking_admin_membership

    # A second revoke of the same, already-revoked grant is a no-op.
    async with tenant_session(ctx) as session:
        already_revoked = await repo.revoke(
            session, ctx, grant_id=grant_id, revoked_by=revoking_admin_membership
        )
        assert already_revoked is False


async def test_grant_created_for_one_tenant_is_invisible_and_unusable_by_another(environment):
    """AC5: a standing grant created for one tenant is invisible to, and unusable by, a second
    tenant's session under the standard RLS policy."""
    seeded_a = await seed_tenant(environment)
    seeded_b = await seed_tenant(environment)
    _, admin_a = await seed_membership(environment, tenant_id=seeded_a.tenant_id, role="admin")
    _, agent_a = await seed_membership(environment, tenant_id=seeded_a.tenant_id, role="agent")
    _, admin_b = await seed_membership(environment, tenant_id=seeded_b.tenant_id, role="admin")

    ctx_a = RequestContext(
        tenant_id=seeded_a.tenant_id, identity_id=uuid.uuid4(), roles=frozenset({"admin"})
    )
    repo = StandingGrantRepository()
    async with tenant_session(ctx_a) as session:
        grant = await repo.create(
            session,
            ctx_a,
            agent_membership_id=agent_a,
            tool_name="send_invoice",
            granted_by=admin_a,
        )
        grant_id = grant.id

    ctx_b = RequestContext(
        tenant_id=seeded_b.tenant_id, identity_id=uuid.uuid4(), roles=frozenset({"admin"})
    )
    async with tenant_session(ctx_b) as session:
        listing_b = await repo.list_for_tenant(session, ctx_b)
        assert listing_b == []

        active_from_b = await repo.get_active(
            session, ctx_b, agent_membership_id=agent_a, tool_name="send_invoice"
        )
        assert active_from_b is None

        revoked_from_b = await repo.revoke(session, ctx_b, grant_id=grant_id, revoked_by=admin_b)
        assert revoked_from_b is False

        # Directly against the table, bypassing the repository's own WHERE clause entirely.
        row = (
            await session.execute(
                text("SELECT id FROM standing_grants WHERE id = :id"), {"id": grant_id}
            )
        ).first()
        assert row is None

    # Still there and active under its own tenant.
    async with tenant_session(ctx_a) as session:
        still_active = await repo.get_active(
            session, ctx_a, agent_membership_id=agent_a, tool_name="send_invoice"
        )
        assert still_active is not None
        assert still_active.id == grant_id


async def test_listing_shows_granting_membership_and_created_at_for_every_grant(environment):
    """AC6: listing a tenant's standing grants shows the granting membership and creation time
    for every grant, active and revoked alike."""
    seeded = await seed_tenant(environment)
    _, admin_membership = await seed_membership(
        environment, tenant_id=seeded.tenant_id, role="admin"
    )
    _, agent_membership = await seed_membership(
        environment, tenant_id=seeded.tenant_id, role="agent"
    )
    ctx = RequestContext(
        tenant_id=seeded.tenant_id, identity_id=uuid.uuid4(), roles=frozenset({"admin"})
    )
    repo = StandingGrantRepository()

    async with tenant_session(ctx) as session:
        active_grant = await repo.create(
            session,
            ctx,
            agent_membership_id=agent_membership,
            tool_name="send_invoice",
            granted_by=admin_membership,
        )
        revoked_grant = await repo.create(
            session,
            ctx,
            agent_membership_id=agent_membership,
            tool_name="delete_document",
            granted_by=admin_membership,
        )
        await repo.revoke(session, ctx, grant_id=revoked_grant.id, revoked_by=admin_membership)

        listing = {record.id: record for record in await repo.list_for_tenant(session, ctx)}

    assert listing[active_grant.id].granted_by == admin_membership
    assert listing[active_grant.id].created_at is not None
    assert listing[active_grant.id].revoked_at is None

    assert listing[revoked_grant.id].granted_by == admin_membership
    assert listing[revoked_grant.id].created_at is not None
    assert listing[revoked_grant.id].revoked_at is not None
    assert listing[revoked_grant.id].revoked_by == admin_membership


async def test_admin_tool_functions_create_revoke_and_list_end_to_end(environment):
    """The tool-layer wrappers (app/tools/standing_grants.py) work end to end for an admin
    caller: create, list, revoke, list again."""
    seeded = await seed_tenant(environment)
    _, admin_membership = await seed_membership(
        environment, tenant_id=seeded.tenant_id, role="admin"
    )
    _, agent_membership = await seed_membership(
        environment, tenant_id=seeded.tenant_id, role="agent"
    )
    ctx = RequestContext(
        tenant_id=seeded.tenant_id, identity_id=uuid.uuid4(), roles=frozenset({"admin"})
    )

    created = await create_standing_grant(
        ctx,
        agent_membership_id=agent_membership,
        tool_name="send_invoice",
        granted_by=admin_membership,
    )
    assert created.revoked_at is None

    listing = await list_standing_grants(ctx)
    assert [record.id for record in listing] == [created.id]

    revoked = await revoke_standing_grant(ctx, grant_id=created.id, revoked_by=admin_membership)
    assert revoked is True

    listing_after = await list_standing_grants(ctx)
    assert listing_after[0].revoked_at is not None
