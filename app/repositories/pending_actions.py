"""Pending actions repository (ADR-0007, Spec 5 / #37): the only path to `pending_actions`.

The session comes from `tenant_session(ctx)` and is therefore tenant-bound; RLS filters every
method to `ctx.tenant_id` before any code here runs (migration 0022) -- a pending action created
under one tenant is simply not a row a second tenant's session can see.

**The one operation that matters most.** `verify()` is what a writing tool calls right before it
actually runs, and it fails closed on every axis rather than guessing:

- the stored record must exist (in this tenant);
- its `status` must be exactly `approved` -- `pending`, `refused`, or an unknown id all refuse;
- its `expires_at` must not have passed, checked against the real wall clock at the moment of the
  call, never a value cached earlier in the request;
- a hash of the tool name and arguments the caller is *about to run*, recomputed fresh right here,
  must match the hash stored when the action was first proposed -- a different call than the one
  a member actually saw and approved refuses, even when the tool name, tenant, and conversation
  all still match.

This mirrors ADR-0007 directly: "the approval is verified against that record, never against the
message the client sends back," and a mismatching hash or an expired record "fails closed."

Because `verify()` requires exactly `approved`, an action that has been claimed for execution
(`executing`) or whose execution has already been recorded (`executed`/`execution_failed`, below)
never verifies again -- a second guarantee against running the same approved call twice,
independent of the hash and expiry checks.

**Verify, then claim, in one transaction (#122).** `verify()` only reads; the caller that is about
to run the tool (`app.tools.approvals.require_approval`) follows a successful `verify()` with
`claim_for_execution()` *in the same session and transaction*, and may run the tool only when the
claim returns True. The claim is a guarded `UPDATE ... WHERE status = 'approved'` that moves the
row to `executing`: two resumes of the same action that both pass `verify()` concurrently both
reach the `UPDATE`, the second blocks on the first's row lock, re-checks the `WHERE` once the
first commits, and updates zero rows -- so exactly one of them may execute, and the other is
refused (`status_executing`). The claim is a separate method rather than a side effect of
`verify()` so that `verify()` stays what its name says -- a read-only check its callers and tests
can run on a row without changing it -- and the one state change lives where its guard is.

**Status lifecycle (#82, migration 0043).** Every transition lives in this class and nothing
outside it writes `status`; each is a single guarded `UPDATE ... WHERE status = <from>`, so a
transition happens at most once even under concurrency (Postgres re-checks the `WHERE` after
taking the row lock -- the loser updates zero rows):

    pending   --resolve()--------------------> approved | refused
    pending   --expire_overdue() (sweep)-----> expired
    approved  --mark_expired() (late resume)-> expired
    approved  --claim_for_execution()--------> executing
    executing --mark_execution_outcome()-----> executed | execution_failed

`expired`, `refused`, `executed`, `execution_failed` are terminal. `executing` is not, but only
`mark_execution_outcome()` ever leaves it: no sweep, timeout, or late resume touches an
`executing` row. A row left there by a process that crashed mid-execution stays there -- a
visible, fail-closed state (it can never run again), deliberately not something this class
guesses about (#122). Every operation except the
sweep's `expire_overdue()` is scoped by the single pending action's own primary key (plus
tenant), not by conversation or tool name, so moving one pending action never touches another --
even one in the same tenant and conversation. `expire_overdue()` is scoped by tenant (RLS plus the
`WHERE` clause) and touches only that tenant's overdue `pending` rows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.context import RequestContext
from app.db.models import PendingAction

PENDING = "pending"
APPROVED = "approved"
REFUSED = "refused"
EXPIRED = "expired"
EXECUTING = "executing"
EXECUTED = "executed"
EXECUTION_FAILED = "execution_failed"

STATUSES: frozenset[str] = frozenset(
    {PENDING, APPROVED, REFUSED, EXPIRED, EXECUTING, EXECUTED, EXECUTION_FAILED}
)


def hash_arguments(tool_name: str, arguments: dict[str, Any]) -> str:
    """A deterministic hex-encoded SHA-256 digest of a tool call: the tool name plus its exact
    arguments, canonicalized (sorted keys, no incidental whitespace) so the same logical call
    always hashes the same way regardless of key order. Not a secret -- this only needs to be
    stable and collision-resistant for equality checking, never verified against an untrusted
    input the way a credential secret is (see `app/repositories/agent_credentials.py`)."""
    canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(f"{tool_name}:{canonical}".encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """The outcome of `verify()`: `ok` is the only thing a caller should branch on; `reason`
    exists for logging/audit, never to let a caller treat a specific failure differently -- every
    failure here refuses the write outright."""

    ok: bool
    reason: str | None = None


def status_reason(status: str) -> str:
    """The one spelling of a `VerificationResult.reason` that names a status (`status_expired`,
    `status_executing`, ...): `verify()` produces it here, and the approval machinery
    (`app.tools.approvals`) compares against and reports the same function's output, so the two
    can never drift apart."""
    return f"status_{status}"


@dataclass(frozen=True, slots=True)
class ExpiredPendingAction:
    """One row `expire_overdue()` moved from `pending` to `expired` -- exactly what the sweep needs
    to write that row's one `expired` audit event (the asking membership is the actor)."""

    id: UUID
    asking_membership_id: UUID
    tool_name: str


class PendingActionRepository:
    async def create(
        self,
        session: AsyncSession,
        ctx: RequestContext,
        *,
        conversation_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: str,
        asking_membership_id: UUID,
        expires_in: timedelta,
    ) -> PendingAction:
        """Writes down a pending action *before* the approval is ever shown to the member
        (ADR-0007's ordering requirement). `expires_in` is the caller's own configured window
        (`Settings.pending_action_expiry_seconds`, or a per-call override) -- this method never
        hard-codes a duration, so two callers configured with different windows really do expire
        at different offsets from their own creation time. `tool_call_id` is the model's own id
        for this call (migration 0038) -- the key `get_by_tool_call()` uses to find this same row
        again on the resumed run."""
        action = PendingAction(
            tenant_id=ctx.tenant_id,
            conversation_id=conversation_id,
            tool_name=tool_name,
            args_hash=hash_arguments(tool_name, arguments),
            tool_call_id=tool_call_id,
            asking_membership_id=asking_membership_id,
            status=PENDING,
            expires_at=datetime.now(UTC) + expires_in,
        )
        session.add(action)
        await session.flush()
        return action

    async def get_by_tool_call(
        self,
        session: AsyncSession,
        ctx: RequestContext,
        *,
        conversation_id: str,
        tool_call_id: str,
    ) -> PendingAction | None:
        """The most recent pending action raised for this exact model tool call, scoped to
        `ctx.tenant_id` and `conversation_id` -- how a resumed run (and the request that resolves
        a member's approve/refuse decision) finds the row `create()` wrote down for it, without
        trusting anything the client sends back for the id itself. None for an unknown
        tool_call_id, or one belonging to another tenant or conversation."""
        return (
            (
                await session.execute(
                    select(PendingAction)
                    .where(
                        PendingAction.tenant_id == ctx.tenant_id,
                        PendingAction.conversation_id == conversation_id,
                        PendingAction.tool_call_id == tool_call_id,
                    )
                    .order_by(PendingAction.created_at.desc())
                )
            )
            .scalars()
            .first()
        )

    async def get(
        self, session: AsyncSession, ctx: RequestContext, *, pending_action_id: UUID
    ) -> PendingAction | None:
        """The pending action by id, scoped to `ctx.tenant_id` (via RLS and, redundantly but
        cheaply, the WHERE clause). None for an unknown id or one belonging to another tenant --
        RLS makes those indistinguishable by construction."""
        return (
            await session.execute(
                select(PendingAction).where(
                    PendingAction.tenant_id == ctx.tenant_id,
                    PendingAction.id == pending_action_id,
                )
            )
        ).scalar_one_or_none()

    async def resolve(
        self,
        session: AsyncSession,
        ctx: RequestContext,
        *,
        pending_action_id: UUID,
        approved: bool,
        resolved_by: UUID,
    ) -> bool:
        """Moves a `pending` action to `approved` or `refused`, once. Scoped by this one action's
        own primary key (plus tenant), so answering it never affects any other pending action --
        including one in the same tenant and the same conversation. Returns False (a no-op) for
        an unknown id, a cross-tenant id, or an action that is no longer `pending`."""
        result = await session.execute(
            update(PendingAction)
            .where(
                PendingAction.tenant_id == ctx.tenant_id,
                PendingAction.id == pending_action_id,
                PendingAction.status == PENDING,
            )
            .values(
                status=APPROVED if approved else REFUSED,
                resolved_at=datetime.now(UTC),
                resolved_by=resolved_by,
            )
        )
        return result.rowcount > 0

    async def mark_execution_outcome(
        self,
        session: AsyncSession,
        ctx: RequestContext,
        *,
        pending_action_id: UUID,
        success: bool,
    ) -> bool:
        """Moves an `executing` action to `executed` (`success=True`) or `execution_failed`, once
        -- called from `app.tools.approvals.record_write_outcome` in the same transaction as the
        matching `executed`/`failed_to_execute` audit row. Returns False (a no-op) for an unknown
        or cross-tenant id, or an action that is not `executing`: one that was never claimed
        (still `approved` -- no path may skip `claim_for_execution()`), or whose outcome is
        already recorded. A standing-grant execution has no pending action and never calls
        this."""
        result = await session.execute(
            update(PendingAction)
            .where(
                PendingAction.tenant_id == ctx.tenant_id,
                PendingAction.id == pending_action_id,
                PendingAction.status == EXECUTING,
            )
            .values(status=EXECUTED if success else EXECUTION_FAILED)
        )
        return result.rowcount > 0

    async def claim_for_execution(
        self, session: AsyncSession, ctx: RequestContext, *, pending_action_id: UUID
    ) -> bool:
        """Moves an `approved` action to `executing`, once (#122) -- the claim a caller must win
        before it runs the tool. Called by `app.tools.approvals.require_approval` right after a
        successful `verify()`, in the *same* session/transaction, so the claim is the atomic
        second half of the verification (module docstring). Returns True only when this call's
        guarded `UPDATE` moved the row; False (a no-op) for an unknown or cross-tenant id, or an
        action that is not `approved` -- `pending`, `refused`, `expired`, already `executing`
        (a concurrent resume won the claim), or `executed`/`execution_failed`. A caller that gets
        False must not execute.

        Deliberately no expiry or hash check here: `verify()` has just made both in the same
        transaction, and repeating them would only open a second, different answer to the same
        question."""
        result = await session.execute(
            update(PendingAction)
            .where(
                PendingAction.tenant_id == ctx.tenant_id,
                PendingAction.id == pending_action_id,
                PendingAction.status == APPROVED,
            )
            .values(status=EXECUTING)
        )
        return result.rowcount > 0

    async def mark_expired(
        self,
        session: AsyncSession,
        ctx: RequestContext,
        *,
        pending_action_id: UUID,
        now: datetime | None = None,
    ) -> bool:
        """The late-resume transition: an `approved` action whose `expires_at` is at or before
        `now` (default: the wall clock -- the same clock `verify()` checks, so the two never
        disagree about whether an action has expired) moves to `expired`, once. The
        caller (`app.tools.approvals.require_approval`) records the `expired` audit event only when
        this returns True -- so a second late resume, or a row the sweep already expired (which is
        no longer `approved`), never produces a second event. Returns False for an unknown or
        cross-tenant id, an action that is not `approved`, or one still inside its window."""
        result = await session.execute(
            update(PendingAction)
            .where(
                PendingAction.tenant_id == ctx.tenant_id,
                PendingAction.id == pending_action_id,
                PendingAction.status == APPROVED,
                PendingAction.expires_at <= (now or datetime.now(UTC)),
            )
            .values(status=EXPIRED)
        )
        return result.rowcount > 0

    async def expire_overdue(
        self, session: AsyncSession, ctx: RequestContext, *, now: datetime | None = None
    ) -> list[ExpiredPendingAction]:
        """The sweep's transition (`app/pending_action_sweep.py`): every `pending` row of this
        tenant whose `expires_at` is at or before `now` (default: the wall clock) moves to
        `expired`, in one statement. Returns exactly the rows it moved -- never a row another
        tenant owns (RLS plus the `WHERE` clause), never a fresh one, never one already resolved.
        A second call finds nothing, which is what makes the sweep idempotent. An overdue
        `approved` row is deliberately left alone: whether it expires is decided when someone
        tries to run it (`mark_expired`), and it is never the sweep's to claim. An `executing`
        row is never touched either, however far past its `expires_at` (#122): only the outcome
        path leaves that state."""
        cutoff = now or datetime.now(UTC)
        rows = await session.execute(
            update(PendingAction)
            .where(
                PendingAction.tenant_id == ctx.tenant_id,
                PendingAction.status == PENDING,
                PendingAction.expires_at <= cutoff,
            )
            .values(status=EXPIRED)
            .returning(
                PendingAction.id, PendingAction.asking_membership_id, PendingAction.tool_name
            )
        )
        return [
            ExpiredPendingAction(
                id=row.id, asking_membership_id=row.asking_membership_id, tool_name=row.tool_name
            )
            for row in rows
        ]

    async def verify(
        self,
        session: AsyncSession,
        ctx: RequestContext,
        *,
        pending_action_id: UUID,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> VerificationResult:
        """Recomputes the argument hash from the call the caller is about to make and checks it,
        the record's status, and its expiry against the real clock -- all three must hold, or
        verification refuses outright. See the module docstring for why each check exists.

        The status check requires exactly `approved`: `status_executing` means another resume
        has already claimed this action and is running it (#122);
        `status_executed`/`status_execution_failed` mean this action already ran (its outcome is
        recorded) and it cannot run again -- a guarantee independent of the hash and the expiry;
        `status_expired` means the sweep (or an earlier late resume) has already expired it and
        written its one `expired` event.

        Read-only: a caller about to execute follows an `ok` result with `claim_for_execution()`
        in the same transaction and runs the tool only if that claim succeeds (module
        docstring)."""
        action = await self.get(session, ctx, pending_action_id=pending_action_id)
        if action is None:
            return VerificationResult(ok=False, reason="not_found")
        if action.status != APPROVED:
            return VerificationResult(ok=False, reason=status_reason(action.status))
        if datetime.now(UTC) >= _as_aware(action.expires_at):
            return VerificationResult(ok=False, reason="expired")
        if action.args_hash != hash_arguments(tool_name, arguments):
            return VerificationResult(ok=False, reason="hash_mismatch")
        return VerificationResult(ok=True)


def _as_aware(value: datetime) -> datetime:
    """asyncpg round-trips `timestamptz` as a naive UTC datetime by default; normalize to
    timezone-aware UTC so comparisons against `datetime.now(UTC)` are always valid."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
