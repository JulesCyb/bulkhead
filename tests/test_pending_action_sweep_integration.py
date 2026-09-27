"""Real-Postgres tests for the pending-action lifecycle beyond approve/refuse (#82, ADR-0007):
the execution outcome recorded on the pending action itself, and the sweep that marks an
unanswered pending action `expired` -- with exactly one `expired` audit event -- even when nobody
ever tries to resume it. Runs against PostgreSQL + pgvector (`pgserver`, via
`uv sync --group dbtest`). Pattern: `tests/test_retention_integration.py` (the job seam) and
`tests/test_rls_integration.py`'s pending-action tests (the repository seam).

Two seams, two halves of this file:

- **Repository** (`PendingActionRepository`): every status transition lives there and nowhere
  else -- `claim_for_execution` moves an `approved` action to `executing` exactly once, even for
  two concurrent claims (#122); `mark_execution_outcome` moves only an `executing` action,
  `mark_expired` only an overdue `approved` one, `expire_overdue` only overdue `pending` rows of
  the calling tenant (never an `executing` one); and `verify()` refuses an action that has been
  claimed or has already executed.
- **Job** (`app.pending_action_sweep.run_pending_action_sweep`): visits every non-suspended
  tenant through `tenant_session(ctx)` exactly like the retention job, is idempotent, and never
  reaches across tenants. The cluster is shared across the whole test session, so every
  assertion here is scoped to the tenants this test seeded itself -- the job also visits every
  other test's tenants, which is harmless and not asserted on.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import create_async_engine

pgserver = pytest.importorskip("pgserver")

from app.context import RequestContext  # noqa: E402
from app.db.session import tenant_session  # noqa: E402
from app.repositories import approval_audit as audit_kinds  # noqa: E402
from app.repositories.approval_audit import ApprovalAuditRepository  # noqa: E402
from app.repositories.pending_actions import PendingActionRepository  # noqa: E402
from tests.support import (  # noqa: E402
    cluster,
    environment,
    seed_conversation,
    seed_tenant,
)

_ = (cluster, environment)

CONVERSATION_ID = "conv-sweep"


async def _tenant_with_conversation(environment, name: str):
    tenant = await seed_tenant(environment, name=name, roles=["member"], via_operator=False)
    await seed_conversation(
        environment,
        tenant_id=tenant.tenant_id,
        identity_id=tenant.identities["member"],
        conversation_id=CONVERSATION_ID,
    )
    return tenant


def _ctx(tenant) -> RequestContext:
    return RequestContext(tenant_id=tenant.tenant_id, identity_id=tenant.identities["member"])


async def _create_action(
    tenant,
    *,
    tool_call_id: str,
    expires_in: timedelta,
    approve: bool = False,
    claim: bool = False,
) -> uuid.UUID:
    """`approve` resolves the new action `approved`; `claim` (which implies `approve`) then claims
    it `executing` (#122) -- the state an action is in while its tool is running."""
    approve = approve or claim
    ctx = _ctx(tenant)
    repo = PendingActionRepository()
    async with tenant_session(ctx) as session:
        action = await repo.create(
            session,
            ctx,
            conversation_id=CONVERSATION_ID,
            tool_name="rename_document",
            arguments={"document_id": tool_call_id},
            tool_call_id=tool_call_id,
            asking_membership_id=tenant.memberships["member"],
            expires_in=expires_in,
        )
        if approve:
            assert await repo.resolve(
                session,
                ctx,
                pending_action_id=action.id,
                approved=True,
                resolved_by=tenant.memberships["member"],
            )
        if claim:
            assert await repo.claim_for_execution(session, ctx, pending_action_id=action.id)
        return action.id


async def _move_expiry_into_the_past(environment, action_id: uuid.UUID) -> None:
    """An action claimed while still inside its window, whose window has since passed -- how an
    `executing` row ends up overdue in real life (a tool body that runs past the expiry, or a
    process that crashed mid-execution). Written as the superuser: a test fixture, not a
    transition the application ever makes."""
    engine = create_async_engine(environment.superuser_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE pending_actions SET expires_at = now() - interval '1 hour' "
                    "WHERE id = :id"
                ),
                {"id": action_id},
            )
    finally:
        await engine.dispose()


async def _status(tenant, action_id: uuid.UUID) -> str | None:
    """Read through the tenant's *own* session -- the RLS-scoped path -- never a superuser."""
    ctx = _ctx(tenant)
    async with tenant_session(ctx) as session:
        action = await PendingActionRepository().get(session, ctx, pending_action_id=action_id)
    return action.status if action is not None else None


async def _audit(tenant, action_id: uuid.UUID):
    ctx = _ctx(tenant)
    async with tenant_session(ctx) as session:
        return await ApprovalAuditRepository().list_for_pending_action(
            session, ctx, pending_action_id=action_id
        )


# --- repository seam -----------------------------------------------------------------------------


async def test_execution_outcome_moves_only_an_executing_action_of_this_tenant(environment):
    """`executing -> executed` / `executing -> execution_failed`, once, scoped by tenant + id: a
    `pending` action, an `approved` one that was never claimed (#122: no path may skip the claim),
    an already-executed one, and another tenant's executing one all refuse."""
    tenant_a = await _tenant_with_conversation(environment, "OutcomeA")
    tenant_b = await _tenant_with_conversation(environment, "OutcomeB")
    repo = PendingActionRepository()
    ctx_a, ctx_b = _ctx(tenant_a), _ctx(tenant_b)

    still_pending = await _create_action(
        tenant_a, tool_call_id="call-pending", expires_in=timedelta(minutes=5)
    )
    unclaimed = await _create_action(
        tenant_a, tool_call_id="call-unclaimed", expires_in=timedelta(minutes=5), approve=True
    )
    succeeded = await _create_action(
        tenant_a, tool_call_id="call-ok", expires_in=timedelta(minutes=5), claim=True
    )
    failed = await _create_action(
        tenant_a, tool_call_id="call-fail", expires_in=timedelta(minutes=5), claim=True
    )

    async with tenant_session(ctx_a) as session:
        assert not await repo.mark_execution_outcome(
            session, ctx_a, pending_action_id=still_pending, success=True
        )
        assert not await repo.mark_execution_outcome(
            session, ctx_a, pending_action_id=unclaimed, success=True
        )
    # Another tenant's session cannot move tenant A's executing action.
    async with tenant_session(ctx_b) as session:
        assert not await repo.mark_execution_outcome(
            session, ctx_b, pending_action_id=succeeded, success=True
        )
    assert await _status(tenant_a, succeeded) == "executing"

    async with tenant_session(ctx_a) as session:
        assert await repo.mark_execution_outcome(
            session, ctx_a, pending_action_id=succeeded, success=True
        )
        assert await repo.mark_execution_outcome(
            session, ctx_a, pending_action_id=failed, success=False
        )
    async with tenant_session(ctx_a) as session:
        # Terminal: a second outcome for the same action changes nothing.
        assert not await repo.mark_execution_outcome(
            session, ctx_a, pending_action_id=succeeded, success=False
        )

    assert await _status(tenant_a, still_pending) == "pending"
    assert await _status(tenant_a, unclaimed) == "approved"
    assert await _status(tenant_a, succeeded) == "executed"
    assert await _status(tenant_a, failed) == "execution_failed"


async def test_two_concurrent_claims_of_one_approved_action_move_it_exactly_once(environment):
    """#122: two sessions that have *both* verified the same `approved` action (a barrier holds
    each one between `verify()` and the claim until both have verified -- exactly the window the
    race lived in) both try to claim it; Postgres re-checks the guarded `UPDATE`'s `WHERE` after
    the winner's row lock is released, so exactly one claim moves the row and the other changes
    nothing."""
    tenant = await _tenant_with_conversation(environment, "ClaimRace")
    ctx = _ctx(tenant)
    action_id = await _create_action(
        tenant, tool_call_id="call-race", expires_in=timedelta(minutes=5), approve=True
    )
    both_verified = asyncio.Barrier(2)

    async def verify_then_claim() -> bool:
        repo = PendingActionRepository()
        async with tenant_session(ctx) as session:
            verification = await repo.verify(
                session,
                ctx,
                pending_action_id=action_id,
                tool_name="rename_document",
                arguments={"document_id": "call-race"},
            )
            assert verification.ok
            await both_verified.wait()
            return await repo.claim_for_execution(session, ctx, pending_action_id=action_id)

    results = await asyncio.gather(verify_then_claim(), verify_then_claim())

    assert sorted(results) == [False, True]
    assert await _status(tenant, action_id) == "executing"


async def test_claim_changes_nothing_unless_the_action_is_approved_in_this_tenant(environment):
    """#122: the claim is a guarded `approved -> executing` move and nothing else -- a `pending`,
    `refused`, `expired`, already-`executing` or `executed` action, and another tenant's
    `approved` one, all change zero rows and keep their status."""
    tenant_a = await _tenant_with_conversation(environment, "ClaimA")
    tenant_b = await _tenant_with_conversation(environment, "ClaimB")
    repo = PendingActionRepository()
    ctx_a, ctx_b = _ctx(tenant_a), _ctx(tenant_b)

    pending = await _create_action(
        tenant_a, tool_call_id="call-pending", expires_in=timedelta(minutes=5)
    )
    refused = await _create_action(
        tenant_a, tool_call_id="call-refused", expires_in=timedelta(minutes=5)
    )
    expired = await _create_action(
        tenant_a, tool_call_id="call-expired", expires_in=timedelta(minutes=-5), approve=True
    )
    executing = await _create_action(
        tenant_a, tool_call_id="call-executing", expires_in=timedelta(minutes=5), claim=True
    )
    executed = await _create_action(
        tenant_a, tool_call_id="call-executed", expires_in=timedelta(minutes=5), claim=True
    )
    foreign_approved = await _create_action(
        tenant_b, tool_call_id="call-foreign", expires_in=timedelta(minutes=5), approve=True
    )
    async with tenant_session(ctx_a) as session:
        assert await repo.resolve(
            session,
            ctx_a,
            pending_action_id=refused,
            approved=False,
            resolved_by=tenant_a.memberships["member"],
        )
        assert await repo.mark_expired(session, ctx_a, pending_action_id=expired)
        assert await repo.mark_execution_outcome(
            session, ctx_a, pending_action_id=executed, success=True
        )

    async with tenant_session(ctx_a) as session:
        for action_id in (pending, refused, expired, executing, executed, foreign_approved):
            assert not await repo.claim_for_execution(session, ctx_a, pending_action_id=action_id)

    assert await _status(tenant_a, pending) == "pending"
    assert await _status(tenant_a, refused) == "refused"
    assert await _status(tenant_a, expired) == "expired"
    assert await _status(tenant_a, executing) == "executing"
    assert await _status(tenant_a, executed) == "executed"
    assert await _status(tenant_b, foreign_approved) == "approved"
    # And the same foreign row, claimed from its own tenant, moves normally.
    async with tenant_session(ctx_b) as session:
        assert await repo.claim_for_execution(session, ctx_b, pending_action_id=foreign_approved)


async def test_verify_refuses_an_action_that_has_been_claimed(environment):
    """#122: a claimed (`executing`) action no longer verifies -- `verify()` reports
    `status_executing`, which is what a late or concurrent resume is refused with."""
    tenant = await _tenant_with_conversation(environment, "VerifyExecuting")
    ctx = _ctx(tenant)
    action_id = await _create_action(
        tenant, tool_call_id="call-claimed", expires_in=timedelta(minutes=5), claim=True
    )

    async with tenant_session(ctx) as session:
        result = await PendingActionRepository().verify(
            session,
            ctx,
            pending_action_id=action_id,
            tool_name="rename_document",
            arguments={"document_id": "call-claimed"},
        )
    assert result.ok is False
    assert result.reason == "status_executing"


async def test_verify_refuses_an_action_that_already_executed(environment):
    """The second, independent guarantee beside hash and expiry: `verify()` requires exactly
    `approved`, so an action whose execution has been recorded can never verify again."""
    tenant = await _tenant_with_conversation(environment, "VerifyExecuted")
    ctx = _ctx(tenant)
    repo = PendingActionRepository()
    action_id = await _create_action(
        tenant, tool_call_id="call-once", expires_in=timedelta(minutes=5), approve=True
    )
    arguments = {"document_id": "call-once"}

    async with tenant_session(ctx) as session:
        first = await repo.verify(
            session,
            ctx,
            pending_action_id=action_id,
            tool_name="rename_document",
            arguments=arguments,
        )
        assert first.ok
        assert await repo.claim_for_execution(session, ctx, pending_action_id=action_id)
        assert await repo.mark_execution_outcome(
            session, ctx, pending_action_id=action_id, success=True
        )
    async with tenant_session(ctx) as session:
        second = await repo.verify(
            session,
            ctx,
            pending_action_id=action_id,
            tool_name="rename_document",
            arguments=arguments,
        )
    assert second.ok is False
    assert second.reason == "status_executed"


async def test_expire_overdue_touches_only_this_tenants_overdue_pending_rows(environment):
    """`expire_overdue` returns exactly the rows it moved (id + asking membership) and moves only
    this tenant's `pending` rows whose `expires_at` has passed -- a fresh pending row, an overdue
    *approved* row (the late-resume path's business, not the sweep's), an overdue *executing* row
    (#122: only the outcome path ever leaves that state), and another tenant's overdue pending row
    are all left alone."""
    tenant_a = await _tenant_with_conversation(environment, "ExpireA")
    tenant_b = await _tenant_with_conversation(environment, "ExpireB")
    repo = PendingActionRepository()
    ctx_a = _ctx(tenant_a)

    overdue = await _create_action(
        tenant_a, tool_call_id="call-overdue", expires_in=timedelta(minutes=-5)
    )
    fresh = await _create_action(tenant_a, tool_call_id="call-fresh", expires_in=timedelta(hours=1))
    overdue_approved = await _create_action(
        tenant_a, tool_call_id="call-approved", expires_in=timedelta(minutes=-5), approve=True
    )
    overdue_executing = await _create_action(
        tenant_a, tool_call_id="call-executing", expires_in=timedelta(hours=1), claim=True
    )
    await _move_expiry_into_the_past(environment, overdue_executing)
    foreign_overdue = await _create_action(
        tenant_b, tool_call_id="call-foreign", expires_in=timedelta(minutes=-5)
    )

    async with tenant_session(ctx_a) as session:
        expired = await repo.expire_overdue(session, ctx_a, now=datetime.now(UTC))

    assert [(e.id, e.asking_membership_id, e.tool_name) for e in expired] == [
        (overdue, tenant_a.memberships["member"], "rename_document")
    ]
    assert await _status(tenant_a, overdue) == "expired"
    assert await _status(tenant_a, fresh) == "pending"
    assert await _status(tenant_a, overdue_approved) == "approved"
    assert await _status(tenant_a, overdue_executing) == "executing"
    assert await _status(tenant_b, foreign_overdue) == "pending"

    # Idempotent: nothing left to expire.
    async with tenant_session(ctx_a) as session:
        assert await repo.expire_overdue(session, ctx_a, now=datetime.now(UTC)) == []


async def test_mark_expired_moves_an_overdue_approved_action_once(environment):
    """The late-resume half: an `approved` action past its expiry moves to `expired` exactly once
    -- the caller records the `expired` audit event only when this returns True, so neither a
    second late resume nor a row the sweep already expired can produce a second event. An
    approved action still inside its window refuses."""
    tenant = await _tenant_with_conversation(environment, "MarkExpired")
    ctx = _ctx(tenant)
    repo = PendingActionRepository()
    overdue = await _create_action(
        tenant, tool_call_id="call-late", expires_in=timedelta(minutes=-5), approve=True
    )
    in_window = await _create_action(
        tenant, tool_call_id="call-in-window", expires_in=timedelta(minutes=5), approve=True
    )

    async with tenant_session(ctx) as session:
        assert await repo.mark_expired(session, ctx, pending_action_id=overdue)
        assert not await repo.mark_expired(session, ctx, pending_action_id=overdue)
        assert not await repo.mark_expired(session, ctx, pending_action_id=in_window)

    assert await _status(tenant, overdue) == "expired"
    assert await _status(tenant, in_window) == "approved"


# --- job seam ------------------------------------------------------------------------------------


async def _run_sweep(environment, **kwargs):
    from app.pending_action_sweep import run_pending_action_sweep

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            return await run_pending_action_sweep(conn, **kwargs)
    finally:
        await engine.dispose()


async def _snapshot(environment, tenant_ids) -> dict:
    """Every pending action's status and every audit row, for the given tenants only -- read as
    the superuser so the snapshot is independent of the code under test."""
    engine = create_async_engine(environment.superuser_url)
    async with engine.connect() as conn:
        actions = (
            await conn.execute(
                text(
                    "SELECT id, status FROM pending_actions WHERE tenant_id = ANY(:tids) "
                    "ORDER BY id"
                ),
                {"tids": list(tenant_ids)},
            )
        ).all()
        audit = (
            await conn.execute(
                text(
                    "SELECT id FROM approval_audit_events WHERE tenant_id = ANY(:tids) ORDER BY id"
                ),
                {"tids": list(tenant_ids)},
            )
        ).all()
    await engine.dispose()
    return {"actions": [tuple(row) for row in actions], "audit": [row.id for row in audit]}


async def test_sweep_expires_each_tenants_overdue_action_once_and_is_idempotent(environment):
    """#82 AC1 + AC2: two tenants, each with an overdue unanswered pending action and a fresh one.
    After one sweep, each overdue row is `expired` with exactly one `expired` audit event (actor:
    the asking membership; no delegation means -- a job acts for nobody), each fresh row is still
    `pending` with no event, and every row and event stays inside its own tenant (read back
    through each tenant's *own* RLS-scoped session). A second sweep changes nothing."""
    tenant_a = await _tenant_with_conversation(environment, "SweepA")
    tenant_b = await _tenant_with_conversation(environment, "SweepB")
    overdue_a = await _create_action(
        tenant_a, tool_call_id="call-a-overdue", expires_in=timedelta(minutes=-5)
    )
    fresh_a = await _create_action(
        tenant_a, tool_call_id="call-a-fresh", expires_in=timedelta(hours=1)
    )
    overdue_b = await _create_action(
        tenant_b, tool_call_id="call-b-overdue", expires_in=timedelta(minutes=-5)
    )
    fresh_b = await _create_action(
        tenant_b, tool_call_id="call-b-fresh", expires_in=timedelta(hours=1)
    )
    # #122: an `executing` row past its expiry (claimed in time, still running or stuck after a
    # crash) is never the sweep's -- no transition, no audit event.
    executing_a = await _create_action(
        tenant_a, tool_call_id="call-a-executing", expires_in=timedelta(hours=1), claim=True
    )
    await _move_expiry_into_the_past(environment, executing_a)

    outcomes = await _run_sweep(environment)

    by_tenant = {o.tenant_id: o for o in outcomes}
    assert by_tenant[tenant_a.tenant_id].expired == (overdue_a,)
    assert by_tenant[tenant_b.tenant_id].expired == (overdue_b,)

    for tenant, overdue, fresh in ((tenant_a, overdue_a, fresh_a), (tenant_b, overdue_b, fresh_b)):
        assert await _status(tenant, overdue) == "expired"
        assert await _status(tenant, fresh) == "pending"
        events = await _audit(tenant, overdue)
        assert [e.kind for e in events] == [audit_kinds.EXPIRED]
        assert events[0].actor_membership_id == tenant.memberships["member"]
        assert events[0].pending_action_id == overdue
        assert events[0].means_kind is None and events[0].means_id is None
        assert await _audit(tenant, fresh) == []
    assert await _status(tenant_a, executing_a) == "executing"
    assert await _audit(tenant_a, executing_a) == []

    # Neither tenant's session can see the other's rows -- the sweep wrote each tenant's event
    # under that tenant's own context, never under the other's.
    assert await _status(tenant_a, overdue_b) is None
    assert await _status(tenant_b, overdue_a) is None
    assert await _audit(tenant_a, overdue_b) == []
    assert await _audit(tenant_b, overdue_a) == []

    before = await _snapshot(environment, [tenant_a.tenant_id, tenant_b.tenant_id])
    second = await _run_sweep(environment)
    after = await _snapshot(environment, [tenant_a.tenant_id, tenant_b.tenant_id])
    assert after == before
    second_by_tenant = {o.tenant_id: o for o in second}
    assert second_by_tenant[tenant_a.tenant_id].expired == ()
    assert second_by_tenant[tenant_b.tenant_id].expired == ()


async def test_sweep_skips_a_suspended_tenant(environment):
    """ADR-0010: nothing is changed for a suspended tenant -- its overdue pending action stays
    `pending` with no event and it is not among the outcomes, while an active tenant in the same
    run is swept normally. (Nothing is lost: once unsuspended the next sweep expires it, and a
    late resume refuses it on its own expiry check meanwhile.)"""
    suspended = await _tenant_with_conversation(environment, "SweepSuspended")
    active = await _tenant_with_conversation(environment, "SweepActive")
    overdue_suspended = await _create_action(
        suspended, tool_call_id="call-suspended", expires_in=timedelta(minutes=-5)
    )
    overdue_active = await _create_action(
        active, tool_call_id="call-active", expires_in=timedelta(minutes=-5)
    )
    await suspended.suspend()

    outcomes = await _run_sweep(environment)

    assert suspended.tenant_id not in {o.tenant_id for o in outcomes}
    assert {o.tenant_id: o.expired for o in outcomes}[active.tenant_id] == (overdue_active,)
    assert await _status(active, overdue_active) == "expired"

    await suspended.unsuspend()
    assert await _status(suspended, overdue_suspended) == "pending"
    assert await _audit(suspended, overdue_suspended) == []

    # Unsuspended, the next sweep catches up.
    outcomes = await _run_sweep(environment)
    assert {o.tenant_id: o.expired for o in outcomes}[suspended.tenant_id] == (overdue_suspended,)


async def test_sweep_never_touches_pending_actions_on_the_enumeration_connection(environment):
    """Like the retention job's AC4: the owner connection passed in only enumerates tenants; every
    write happens on the tenant's own `tenant_session(ctx)`."""
    tenant = await _tenant_with_conversation(environment, "SweepEnumeration")
    overdue = await _create_action(
        tenant, tool_call_id="call-enum", expires_in=timedelta(minutes=-5)
    )

    from app.pending_action_sweep import run_pending_action_sweep

    engine = create_async_engine(environment.owner_url)
    statements: list[str] = []

    def _capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", _capture)
    try:
        async with engine.begin() as conn:
            await run_pending_action_sweep(conn)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", _capture)
        await engine.dispose()

    assert any("enumerate_tenants" in statement.lower() for statement in statements)
    for statement in statements:
        assert "pending_actions" not in statement.lower()
        assert "approval_audit_events" not in statement.lower()
    assert await _status(tenant, overdue) == "expired"


class _FakeSecret:
    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value


class _FakeMigrationSettings:
    def __init__(self, url: str) -> None:
        self.database_url_migrations = _FakeSecret(url)


async def test_sweep_script_entry_point_runs_end_to_end(environment, monkeypatch):
    """`uv run python scripts/sweep_pending_actions.py`, through its real async entry point --
    only `get_migration_settings` is faked (pattern: the retention script's own test)."""
    import scripts.sweep_pending_actions as sweep_script

    tenant = await _tenant_with_conversation(environment, "SweepScript")
    overdue = await _create_action(
        tenant, tool_call_id="call-script", expires_in=timedelta(minutes=-5)
    )
    monkeypatch.setattr(
        sweep_script,
        "get_migration_settings",
        lambda: _FakeMigrationSettings(environment.owner_url),
    )

    await sweep_script._main()

    assert await _status(tenant, overdue) == "expired"
    assert [e.kind for e in await _audit(tenant, overdue)] == [audit_kinds.EXPIRED]
