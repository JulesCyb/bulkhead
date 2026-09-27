"""The pending-action sweep (ADR-0007, #82): an unanswered pending action leaves a record.

Before this job, an `expired` audit event was written only when someone tried to *resume* an
expired pending action (`app/tools/approvals.py`) -- a proposal nobody ever answered never got
one. The sweep closes that gap: for every tenant, every `pending` row whose `expires_at` has
passed moves to `expired` (`PendingActionRepository.expire_overdue`) and gets exactly one
`expired` audit event, both in the same transaction, so a crash can never leave an expired row
without its event or an event without its row. The actor is the row's own asking membership --
whose proposal lapsed -- and the means columns stay null: the job acts on nobody's behalf.

**Idempotent by construction.** An expired row is no longer `pending`, so a second sweep finds
nothing and writes nothing. The late-resume path cannot double-record either: it only ever moves
an *approved* row (`mark_expired`), never a pending one, and records nothing new for a row the
sweep already expired (`app.tools.approvals._audit_kind_for_failed_verification`). An overdue
row a member approved just before its expiry is deliberately not the sweep's to claim -- the
member's own resume decides it (`expired`, or never executed).

**Per tenant, under RLS, like retention.** The loop -- enumerate on the owner connection, build
each tenant's record, skip a suspended tenant with one log line, open `tenant_session(ctx)` as
the job identity -- is `app.tenant_jobs.visit_active_tenants`, the same one the retention job
(`app/retention.py`) uses. The owner connection passed in is used only for that enumeration.

**Suspended tenants are skipped** (ADR-0010: nothing is changed for a suspended tenant). Nothing
is lost by skipping: a late resume still refuses an expired action on its own expiry check, and
the first sweep after unsuspension expires and records whatever lapsed meanwhile.

**Its own script, not a step of the retention job.** The issue suggested "ideally a step of the
same scheduled job"; the two are kept separate (`scripts/sweep_pending_actions.py` beside
`scripts/retention.py`) because their natural schedules differ by three orders of magnitude --
pending actions expire after minutes (`PENDING_ACTION_EXPIRY_SECONDS`), conversations after days
(`retention_days`). Folding the sweep into retention would either run a destructive, days-grained
deletion every few minutes or leave an unanswered proposal unrecorded for up to a day. What the
two jobs genuinely share -- the per-tenant loop -- is shared as code instead of as a schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncConnection

from app.db.session import tenant_session
from app.repositories import approval_audit as audit_kinds
from app.repositories.approval_audit import ApprovalAuditRepository
from app.repositories.pending_actions import PendingActionRepository
from app.tenant_jobs import visit_active_tenants


@dataclass(frozen=True, slots=True)
class SweepOutcome:
    """One tenant's result from a single sweep: the ids of the pending actions it expired (and
    wrote one `expired` event for), in the order the database returned them. Empty when the
    tenant had nothing overdue."""

    tenant_id: UUID
    tenant_name: str
    expired: tuple[UUID, ...]


async def run_pending_action_sweep(
    conn: AsyncConnection, *, now: datetime | None = None
) -> list[SweepOutcome]:
    """Visits every non-suspended tenant (`conn`, an `app_owner` connection, used only to
    enumerate them) and, in that tenant's own `tenant_session(ctx)`, expires its overdue `pending`
    actions and records one `expired` audit event for each (module docstring). `now` (default:
    the wall clock) is the expiry cutoff. Returns one `SweepOutcome` per tenant visited, in the
    order `list_tenants` returns them."""
    outcomes: list[SweepOutcome] = []
    async for visit in visit_active_tenants(conn, job="pending-action sweep"):
        ctx = visit.ctx
        async with tenant_session(ctx) as session:
            expired = await PendingActionRepository().expire_overdue(session, ctx, now=now)
            for action in expired:
                await ApprovalAuditRepository().record(
                    session,
                    ctx,
                    kind=audit_kinds.EXPIRED,
                    tool_name=action.tool_name,
                    actor_membership_id=action.asking_membership_id,
                    pending_action_id=action.id,
                    details={"reason": "expired", "detected_by": "sweep"},
                )
        outcomes.append(
            SweepOutcome(
                tenant_id=visit.tenant.tenant_id,
                tenant_name=visit.tenant.name,
                expired=tuple(action.id for action in expired),
            )
        )
    return outcomes


__all__ = ["SweepOutcome", "run_pending_action_sweep"]
