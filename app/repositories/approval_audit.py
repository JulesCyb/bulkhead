"""Approval audit repository (ADR-0007, Spec 5 / #39): the only path to
`approval_audit_events`.

The session comes from `tenant_session(ctx)` and is therefore tenant-bound; RLS filters every
method to `ctx.tenant_id` before any code here runs (migration 0035) -- an audit record written
for one tenant is simply not a row a second tenant's session can see, by construction.

**Append-only is a property of what can be called.** This module exposes exactly one write
method, `record()`, which always `INSERT`s a new row -- there is no `update`/`delete`/`resolve`
method anywhere in this class, and the `app` role itself holds no `UPDATE`/`DELETE` grant on the
table (migration 0035), so the guarantee holds even against code that bypasses this repository
entirely and talks to the table directly. A milestone, once written, cannot be amended or
removed, only ever added to.

The seven milestone kinds the approval mechanism can reach (ADR-0007, ticket #39) are the module
constants below and nothing else -- `record()` refuses any other string outright, the same
fail-closed posture `PendingActionRepository.verify()` takes for its own checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.context import RequestContext
from app.db.models import ApprovalAuditEvent

REQUESTED = "requested"
APPROVED = "approved"
REFUSED = "refused"
EXPIRED = "expired"
DENIED_FOR_LACK_OF_GRANT = "denied_for_lack_of_grant"
EXECUTED = "executed"
FAILED_TO_EXECUTE = "failed_to_execute"

KINDS: frozenset[str] = frozenset(
    {REQUESTED, APPROVED, REFUSED, EXPIRED, DENIED_FOR_LACK_OF_GRANT, EXECUTED, FAILED_TO_EXECUTE}
)


class InvalidAuditEventKind(ValueError):
    """Raised by `record()` for any `kind` outside `KINDS` -- the seven milestones the approval
    mechanism (ADR-0007) can reach, and nothing else."""


@dataclass(frozen=True, slots=True)
class ApprovalAuditRecord:
    id: UUID
    kind: str
    tool_name: str
    actor_membership_id: UUID
    pending_action_id: UUID | None
    standing_grant_id: UUID | None
    details: dict[str, Any]
    created_at: datetime


def _to_record(event: ApprovalAuditEvent) -> ApprovalAuditRecord:
    return ApprovalAuditRecord(
        id=event.id,
        kind=event.kind,
        tool_name=event.tool_name,
        actor_membership_id=event.actor_membership_id,
        pending_action_id=event.pending_action_id,
        standing_grant_id=event.standing_grant_id,
        details=event.details,
        created_at=event.created_at,
    )


class ApprovalAuditRepository:
    async def record(
        self,
        session: AsyncSession,
        ctx: RequestContext,
        *,
        kind: str,
        tool_name: str,
        actor_membership_id: UUID,
        pending_action_id: UUID | None = None,
        standing_grant_id: UUID | None = None,
        details: dict[str, Any] | None = None,
    ) -> ApprovalAuditEvent:
        """Writes one append-only milestone. `kind` must be one of the seven module constants
        (`KINDS`) -- anything else raises `InvalidAuditEventKind` before any row is written, the
        same fail-closed shape `PendingActionRepository.verify()` uses. `actor_membership_id`
        names the membership the milestone is about (the asking/approving/refusing member, or
        the agent identity's own membership for an autonomous attempt); `pending_action_id`/
        `standing_grant_id` name the specific record behind it, where one exists."""
        if kind not in KINDS:
            raise InvalidAuditEventKind(f"unknown approval audit event kind: {kind!r}")

        event = ApprovalAuditEvent(
            tenant_id=ctx.tenant_id,
            kind=kind,
            tool_name=tool_name,
            actor_membership_id=actor_membership_id,
            pending_action_id=pending_action_id,
            standing_grant_id=standing_grant_id,
            details=details or {},
        )
        session.add(event)
        await session.flush()
        return event

    async def list_for_pending_action(
        self, session: AsyncSession, ctx: RequestContext, *, pending_action_id: UUID
    ) -> list[ApprovalAuditRecord]:
        """Every milestone recorded against one pending action, in this tenant, most recent
        first -- the read this ticket's last acceptance criterion asks for."""
        rows = (
            await session.execute(
                select(ApprovalAuditEvent)
                .where(
                    ApprovalAuditEvent.tenant_id == ctx.tenant_id,
                    ApprovalAuditEvent.pending_action_id == pending_action_id,
                )
                .order_by(ApprovalAuditEvent.seq.desc())
            )
        ).scalars()
        return [_to_record(row) for row in rows]

    async def list_for_standing_grant(
        self, session: AsyncSession, ctx: RequestContext, *, standing_grant_id: UUID
    ) -> list[ApprovalAuditRecord]:
        """Every milestone recorded against one standing grant, in this tenant, most recent
        first."""
        rows = (
            await session.execute(
                select(ApprovalAuditEvent)
                .where(
                    ApprovalAuditEvent.tenant_id == ctx.tenant_id,
                    ApprovalAuditEvent.standing_grant_id == standing_grant_id,
                )
                .order_by(ApprovalAuditEvent.seq.desc())
            )
        ).scalars()
        return [_to_record(row) for row in rows]
