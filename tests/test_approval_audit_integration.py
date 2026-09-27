"""Real isolation test for the approval audit trail (ADR-0007, Spec 5 / #39) against PostgreSQL
+ pgvector (`pgserver`, via `uv sync --group dbtest`). Pattern:
`tests/test_standing_grants_integration.py`.

Proves what a unit test on `ApprovalAuditRepository` alone cannot: that writing one record for
each of the seven milestone kinds really succeeds against live Postgres, that the RLS policy from
migration 0035 isolates one tenant's audit trail from another's, that the `app` role's own grants
(SELECT, INSERT -- no UPDATE/DELETE) make append-only a database-enforced fact and not just a
convention, that a tenant-scoped read of one pending action's or standing grant's milestones comes
back most-recent-first, and (migration 0042, #117) that `record()` persists the delegation means
it reads off `ctx.means` -- or nulls, for a context that carries none -- as its own two columns,
distinct from the approval means (`pending_action_id`/`standing_grant_id`) already covered above.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine

from app.context import RequestContext  # noqa: E402
from app.db.session import tenant_session  # noqa: E402
from app.repositories.approval_audit import (  # noqa: E402
    APPROVED,
    DENIED_FOR_LACK_OF_GRANT,
    EXECUTED,
    EXPIRED,
    FAILED_TO_EXECUTE,
    KINDS,
    REFUSED,
    REQUESTED,
    ApprovalAuditRepository,
    InvalidAuditEventKind,
)

pgserver = pytest.importorskip("pgserver")

from tests.support import cluster, environment, seed_membership, seed_tenant  # noqa: E402

_ = (cluster, environment)


async def test_all_seven_milestone_kinds_can_be_recorded(environment):
    """AC1: writing one audit record for each of the seven milestone kinds succeeds, and each row
    names the acting membership and, where applicable, the specific pending action or standing
    grant it came from."""
    assert KINDS == {
        REQUESTED,
        APPROVED,
        REFUSED,
        EXPIRED,
        DENIED_FOR_LACK_OF_GRANT,
        EXECUTED,
        FAILED_TO_EXECUTE,
    }

    tenant = await seed_tenant(environment, via_operator=False)
    _, member_membership = await seed_membership(
        environment, tenant_id=tenant.tenant_id, role="member"
    )
    _, agent_membership = await seed_membership(
        environment, tenant_id=tenant.tenant_id, role="agent"
    )
    ctx = RequestContext(
        tenant_id=tenant.tenant_id, identity_id=uuid.uuid4(), roles=frozenset({"member"})
    )
    repo = ApprovalAuditRepository()
    pending_action_id = uuid.uuid4()
    standing_grant_id = uuid.uuid4()

    async with tenant_session(ctx) as session:
        for kind, actor, pa_id, sg_id in (
            (REQUESTED, member_membership, pending_action_id, None),
            (APPROVED, member_membership, pending_action_id, None),
            (REFUSED, member_membership, pending_action_id, None),
            (EXPIRED, member_membership, pending_action_id, None),
            (DENIED_FOR_LACK_OF_GRANT, agent_membership, None, None),
            (EXECUTED, agent_membership, None, standing_grant_id),
            (FAILED_TO_EXECUTE, agent_membership, None, standing_grant_id),
        ):
            record = await repo.record(
                session,
                ctx,
                kind=kind,
                tool_name="send_invoice",
                actor_membership_id=actor,
                pending_action_id=pa_id,
                standing_grant_id=sg_id,
            )
            assert record.kind == kind
            assert record.actor_membership_id == actor
            assert record.pending_action_id == pa_id
            assert record.standing_grant_id == sg_id


async def test_recording_an_unknown_kind_is_refused(environment):
    """`record()` refuses any kind outside the seven milestones before writing a row."""
    tenant = await seed_tenant(environment, via_operator=False)
    _, member_membership = await seed_membership(
        environment, tenant_id=tenant.tenant_id, role="member"
    )
    ctx = RequestContext(
        tenant_id=tenant.tenant_id, identity_id=uuid.uuid4(), roles=frozenset({"member"})
    )
    repo = ApprovalAuditRepository()

    async with tenant_session(ctx) as session:
        with pytest.raises(InvalidAuditEventKind):
            await repo.record(
                session,
                ctx,
                kind="always_allow",
                tool_name="send_invoice",
                actor_membership_id=member_membership,
            )


async def test_audit_record_for_one_tenant_is_invisible_to_another(environment):
    """AC2: an audit record written for one tenant is invisible to a second tenant's session
    under the standard RLS policy."""
    tenant_a = await seed_tenant(environment, via_operator=False)
    tenant_b = await seed_tenant(environment, via_operator=False)
    _, member_a = await seed_membership(environment, tenant_id=tenant_a.tenant_id, role="member")

    ctx_a = RequestContext(
        tenant_id=tenant_a.tenant_id, identity_id=uuid.uuid4(), roles=frozenset({"member"})
    )
    repo = ApprovalAuditRepository()
    pending_action_id = uuid.uuid4()
    async with tenant_session(ctx_a) as session:
        event = await repo.record(
            session,
            ctx_a,
            kind=REQUESTED,
            tool_name="send_invoice",
            actor_membership_id=member_a,
            pending_action_id=pending_action_id,
        )
        event_id = event.id

    ctx_b = RequestContext(
        tenant_id=tenant_b.tenant_id, identity_id=uuid.uuid4(), roles=frozenset({"member"})
    )
    async with tenant_session(ctx_b) as session:
        listing_b = await repo.list_for_pending_action(
            session, ctx_b, pending_action_id=pending_action_id
        )
        assert listing_b == []

        # Directly against the table, bypassing the repository's own WHERE clause entirely.
        row = (
            await session.execute(
                text("SELECT id FROM approval_audit_events WHERE id = :id"), {"id": event_id}
            )
        ).first()
        assert row is None

    # Still visible, under its own tenant.
    async with tenant_session(ctx_a) as session:
        listing_a = await repo.list_for_pending_action(
            session, ctx_a, pending_action_id=pending_action_id
        )
        assert [record.id for record in listing_a] == [event_id]


async def test_app_role_has_no_update_or_delete_grant_on_the_audit_table(environment):
    """AC3: the repository exposes no update/delete method (see the module -- `record()`,
    `list_for_pending_action()`, `list_for_standing_grant()` and nothing else), and the `app`
    role itself cannot update or delete a row even bypassing the repository entirely --
    append-only is a property of what can be called, not a convention to remember."""
    assert not hasattr(ApprovalAuditRepository, "update")
    assert not hasattr(ApprovalAuditRepository, "delete")

    tenant = await seed_tenant(environment, via_operator=False)
    _, member_membership = await seed_membership(
        environment, tenant_id=tenant.tenant_id, role="member"
    )
    ctx = RequestContext(
        tenant_id=tenant.tenant_id, identity_id=uuid.uuid4(), roles=frozenset({"member"})
    )
    repo = ApprovalAuditRepository()

    async with tenant_session(ctx) as session:
        event = await repo.record(
            session,
            ctx,
            kind=REQUESTED,
            tool_name="send_invoice",
            actor_membership_id=member_membership,
        )
        event_id = event.id

    app_engine = create_async_engine(environment.app_url)
    async with app_engine.connect() as conn:
        async with conn.begin():
            await conn.execute(text(f"SET LOCAL app.tenant_id = '{tenant.tenant_id}'"))
            with pytest.raises(DBAPIError, match="permission denied"):
                await conn.execute(
                    text("UPDATE approval_audit_events SET kind = 'approved' WHERE id = :id"),
                    {"id": event_id},
                )

    async with app_engine.connect() as conn:
        async with conn.begin():
            await conn.execute(text(f"SET LOCAL app.tenant_id = '{tenant.tenant_id}'"))
            with pytest.raises(DBAPIError, match="permission denied"):
                await conn.execute(
                    text("DELETE FROM approval_audit_events WHERE id = :id"), {"id": event_id}
                )
    await app_engine.dispose()


async def test_read_for_one_pending_action_orders_most_recent_first(environment):
    """AC4: a tenant-scoped read returns the milestones for one pending action in an order that
    makes the most recent one easy to find (most recent first)."""
    tenant = await seed_tenant(environment, via_operator=False)
    _, member_membership = await seed_membership(
        environment, tenant_id=tenant.tenant_id, role="member"
    )
    ctx = RequestContext(
        tenant_id=tenant.tenant_id, identity_id=uuid.uuid4(), roles=frozenset({"member"})
    )
    repo = ApprovalAuditRepository()
    pending_action_id = uuid.uuid4()

    async with tenant_session(ctx) as session:
        first = await repo.record(
            session,
            ctx,
            kind=REQUESTED,
            tool_name="send_invoice",
            actor_membership_id=member_membership,
            pending_action_id=pending_action_id,
        )
        second = await repo.record(
            session,
            ctx,
            kind=APPROVED,
            tool_name="send_invoice",
            actor_membership_id=member_membership,
            pending_action_id=pending_action_id,
        )
        third = await repo.record(
            session,
            ctx,
            kind=EXECUTED,
            tool_name="send_invoice",
            actor_membership_id=member_membership,
            pending_action_id=pending_action_id,
        )

        listing = await repo.list_for_pending_action(
            session, ctx, pending_action_id=pending_action_id
        )

    assert [record.id for record in listing] == [third.id, second.id, first.id]


async def test_read_for_one_standing_grant_orders_most_recent_first(environment):
    """The equivalent read for a standing grant's own milestones, most recent first."""
    tenant = await seed_tenant(environment, via_operator=False)
    _, agent_membership = await seed_membership(
        environment, tenant_id=tenant.tenant_id, role="agent"
    )
    ctx = RequestContext(
        tenant_id=tenant.tenant_id, identity_id=uuid.uuid4(), roles=frozenset({"agent"})
    )
    repo = ApprovalAuditRepository()
    standing_grant_id = uuid.uuid4()

    async with tenant_session(ctx) as session:
        first = await repo.record(
            session,
            ctx,
            kind=EXECUTED,
            tool_name="send_invoice",
            actor_membership_id=agent_membership,
            standing_grant_id=standing_grant_id,
        )
        second = await repo.record(
            session,
            ctx,
            kind=FAILED_TO_EXECUTE,
            tool_name="send_invoice",
            actor_membership_id=agent_membership,
            standing_grant_id=standing_grant_id,
        )

        listing = await repo.list_for_standing_grant(
            session, ctx, standing_grant_id=standing_grant_id
        )

    assert [record.id for record in listing] == [second.id, first.id]


# --- #117: the delegation means (ADR-0005), read from `ctx.means`, distinct from the approval
# means (`pending_action_id`/`standing_grant_id`) already proven above -----------------------------


async def test_record_persists_the_delegation_means_when_the_context_carries_one(environment):
    """`record()` reads `ctx.means` itself -- built here through `SeededTenant.ctx(role,
    means=...)`, the same helper a real HTTP/MCP-resolved context would have gone through -- and
    persists it as `means_kind`/`means_id`, distinct from (and set alongside) the approval means."""
    tenant = await seed_tenant(environment, via_operator=False, roles=["member"])
    member_membership = tenant.memberships["member"]
    ctx = tenant.ctx("member", means=("agent", "assistant"))
    repo = ApprovalAuditRepository()
    pending_action_id = uuid.uuid4()

    async with tenant_session(ctx) as session:
        record = await repo.record(
            session,
            ctx,
            kind=REQUESTED,
            tool_name="send_invoice",
            actor_membership_id=member_membership,
            pending_action_id=pending_action_id,
        )

    assert record.means_kind == "agent"
    assert record.means_id == "assistant"
    assert record.pending_action_id == pending_action_id  # the approval means, untouched


async def test_record_persists_a_credential_delegation_means(environment):
    """The other `MeansKind` (ADR-0005): an agent identity acting on its own credential."""
    tenant = await seed_tenant(environment, via_operator=False, roles=["agent"])
    agent_membership = tenant.memberships["agent"]
    ctx = tenant.ctx("agent", means=("credential", "cred_abc123"))
    repo = ApprovalAuditRepository()

    async with tenant_session(ctx) as session:
        record = await repo.record(
            session,
            ctx,
            kind=EXECUTED,
            tool_name="send_invoice",
            actor_membership_id=agent_membership,
        )

    assert record.means_kind == "credential"
    assert record.means_id == "cred_abc123"


async def test_record_persists_null_means_when_the_context_carries_none(environment):
    """A context built without `means=` (a test, the `stdio` development fallback) records both
    columns null -- allowed, not an error."""
    tenant = await seed_tenant(environment, via_operator=False, roles=["member"])
    member_membership = tenant.memberships["member"]
    ctx = tenant.ctx("member")
    repo = ApprovalAuditRepository()

    async with tenant_session(ctx) as session:
        record = await repo.record(
            session,
            ctx,
            kind=REQUESTED,
            tool_name="send_invoice",
            actor_membership_id=member_membership,
        )

    assert record.means_kind is None
    assert record.means_id is None


async def test_list_for_pending_action_exposes_the_persisted_means(environment):
    """`ApprovalAuditRecord` returned by `list_for_pending_action` carries the same
    `means_kind`/`means_id` the row was written with."""
    tenant = await seed_tenant(environment, via_operator=False, roles=["member"])
    member_membership = tenant.memberships["member"]
    ctx = tenant.ctx("member", means=("agent", "assistant"))
    repo = ApprovalAuditRepository()
    pending_action_id = uuid.uuid4()

    async with tenant_session(ctx) as session:
        await repo.record(
            session,
            ctx,
            kind=REQUESTED,
            tool_name="send_invoice",
            actor_membership_id=member_membership,
            pending_action_id=pending_action_id,
        )
        listing = await repo.list_for_pending_action(
            session, ctx, pending_action_id=pending_action_id
        )

    assert len(listing) == 1
    assert listing[0].means_kind == "agent"
    assert listing[0].means_id == "assistant"


async def test_list_for_standing_grant_exposes_the_persisted_means(environment):
    """The same, for `list_for_standing_grant`."""
    tenant = await seed_tenant(environment, via_operator=False, roles=["agent"])
    agent_membership = tenant.memberships["agent"]
    ctx = tenant.ctx("agent", means=("credential", "cred_xyz789"))
    repo = ApprovalAuditRepository()
    standing_grant_id = uuid.uuid4()

    async with tenant_session(ctx) as session:
        await repo.record(
            session,
            ctx,
            kind=EXECUTED,
            tool_name="send_invoice",
            actor_membership_id=agent_membership,
            standing_grant_id=standing_grant_id,
        )
        listing = await repo.list_for_standing_grant(
            session, ctx, standing_grant_id=standing_grant_id
        )

    assert len(listing) == 1
    assert listing[0].means_kind == "credential"
    assert listing[0].means_id == "cred_xyz789"
