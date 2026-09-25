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

`resolve()` is the one mutation this table allows after creation: a `pending` action moves to
`approved` or `refused`, once, and records who resolved it and when. Resolving one pending action
never touches another -- even one in the same tenant and conversation -- because every operation
here is scoped by the single pending action's own primary key, not by conversation or tool name.
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


class PendingActionRepository:
    async def create(
        self,
        session: AsyncSession,
        ctx: RequestContext,
        *,
        conversation_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        asking_membership_id: UUID,
        expires_in: timedelta,
    ) -> PendingAction:
        """Writes down a pending action *before* the approval is ever shown to the member
        (ADR-0007's ordering requirement). `expires_in` is the caller's own configured window
        (`Settings.pending_action_expiry_seconds`, or a per-call override) -- this method never
        hard-codes a duration, so two callers configured with different windows really do expire
        at different offsets from their own creation time."""
        action = PendingAction(
            tenant_id=ctx.tenant_id,
            conversation_id=conversation_id,
            tool_name=tool_name,
            args_hash=hash_arguments(tool_name, arguments),
            asking_membership_id=asking_membership_id,
            status=PENDING,
            expires_at=datetime.now(UTC) + expires_in,
        )
        session.add(action)
        await session.flush()
        return action

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
        verification refuses outright. See the module docstring for why each check exists."""
        action = await self.get(session, ctx, pending_action_id=pending_action_id)
        if action is None:
            return VerificationResult(ok=False, reason="not_found")
        if action.status != APPROVED:
            return VerificationResult(ok=False, reason=f"status_{action.status}")
        if datetime.now(UTC) >= _as_aware(action.expires_at):
            return VerificationResult(ok=False, reason="expired")
        if action.args_hash != hash_arguments(tool_name, arguments):
            return VerificationResult(ok=False, reason="hash_mismatch")
        return VerificationResult(ok=True)


def _as_aware(value: datetime) -> datetime:
    """asyncpg round-trips `timestamptz` as a naive UTC datetime by default; normalize to
    timezone-aware UTC so comparisons against `datetime.now(UTC)` are always valid."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
