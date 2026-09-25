"""Shared approval machinery behind ADR-0007's writing-tool rule (Spec 5 / #40): the one worked
example a developer deriving this template copies for their own first writing tool.

Every writing tool `chat_assistant` registers (`app/agents/assistant.py`) wraps its tool function
with `args_validator=require_approval`-shaped machinery from this module rather than doing its
own approval bookkeeping -- the tool function itself (`app/tools/documents.py::rename_document`)
stays exactly as thin as a reading tool, going through the repository layer and nothing else.

**Two entry points, two different callers.**

- `require_approval()` is an `args_validator` (pydantic-ai calls it with the same arguments as
  the tool itself, plus `ctx`). It runs *twice* for a member-proposed write: once before any
  approval exists (`ctx.tool_call_approved` is False) and once more on the resumed run after a
  member approved (`ctx.tool_call_approved` is True) -- see the pydantic-ai deferred-tools
  reference this ADR names. It also carries the one branch ADR-0007 keeps in the same mechanism
  even though its own ASGI tests belong to a later ticket (#42): an `agent`-role membership (no
  person present) may act only under an active standing grant for this exact tool, never through
  a pending action at all.
- `resolve_incoming_decisions()` is called by `app/api/chat.py` itself, *before* the agent run,
  for every approve/refuse decision the resumed request's own body carries
  (`VercelAIAdapter.deferred_tool_results`). This exists because a refusal is resolved by
  pydantic-ai substituting `ToolDenied` directly -- the tool's own `args_validator` and body never
  run for a denied call (see `pydantic_ai._tool_execution.build_tool_return_part`), so nothing
  inside the tool can ever record that milestone. Recording "approved" here too (not just
  "refused") keeps both halves of one decision in the same place instead of splitting the member's
  own choice across two different modules.

**What each pass does and does not decide.** `require_approval()`'s first pass never approves
anything by itself -- it only ever writes a `pending` row and raises `ApprovalRequired()` (or,
for an agent membership, denies outright for lack of a grant). Its second pass never trusts that
the resumed run is being approved *because* it is a second pass: it re-verifies the stored pending
action's status, hash, and expiry (`PendingActionRepository.verify()`) and re-reads the acting
membership's *current* role from the database, fresh, both independently of whatever
`resolve_incoming_decisions()` already recorded for the same request. A membership downgraded
between the two passes is refused here, in addition to (not instead of) whatever check ran when
the write was first proposed -- exactly the ordering ADR-0007 and #40's acceptance criteria ask
for.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic_ai import ApprovalRequired, RunContext, ToolFailed

from app.config import get_settings
from app.context import RequestContext
from app.db.session import tenant_session
from app.repositories import approval_audit as audit_kinds
from app.repositories.approval_audit import ApprovalAuditRepository
from app.repositories.memberships import MembershipRepository
from app.repositories.pending_actions import PendingActionRepository
from app.repositories.standing_grants import StandingGrantRepository

if TYPE_CHECKING:
    from app.agents.assistant import AssistantDeps

# Only these two roles may actually execute a writing tool, re-checked fresh at execution time
# (ADR-0007) -- `support` may propose nothing here, and `agent` is handled entirely through the
# standing-grant branch below, never through a pending action.
EXECUTION_ROLES: frozenset[str] = frozenset({"admin", "member"})


class ApprovalDenied(ToolFailed):
    """A terminal, model-visible refusal: nothing executes. Raised by `require_approval()` for
    every failure mode it detects (no membership, no standing grant, a failed verification, a
    role no longer permitted) -- the model sees a failed tool result and can tell the member why,
    but the retry budget is not spent and the write never happens."""


@dataclass(frozen=True, slots=True)
class ApprovalContext:
    """What `require_approval()` learned while deciding a call may run -- handed to the tool body
    (via `AssistantDeps.pending_approval`, set by `require_approval()` just before it returns
    normally) so the tool can record its own `executed`/`failed_to_execute` milestone once it
    knows whether the actual write succeeded. Never constructed by a tool itself."""

    actor_membership_id: UUID
    tool_name: str
    pending_action_id: UUID | None = None
    standing_grant_id: UUID | None = None


async def require_approval(ctx: RunContext[AssistantDeps], **arguments: Any) -> None:
    """The `args_validator` every writing tool registers alongside itself. See the module
    docstring for the two-pass shape and the standing-grant branch.

    Every branch below does its database work and lets the `async with tenant_session(...)` block
    end *normally* before this function raises anything -- `tenant_session` only commits on a
    clean exit and rolls back on an exception escaping it (its own docstring: "Commit at the end,
    rollback on error"), so raising `ApprovalRequired`/`ApprovalDenied` from *inside* that block
    would silently roll back the very pending action or audit record the raise is supposed to
    leave behind. `denial`/`defer` below exist so every write happens, the block commits, and only
    then does this function raise -- outside the block, once the write is durable.
    """
    rc: RequestContext = ctx.deps.ctx
    tool_name = ctx.tool_name or ""
    conversation_id = ctx.deps.conversation_id or ""
    tool_call_id = ctx.tool_call_id or ""

    denial: str | None = None
    defer = False

    async with tenant_session(rc) as session:
        membership = await MembershipRepository().get_by_identity(
            session, rc, identity_id=rc.identity_id
        )
        if membership is None:
            denial = "no membership found for the acting identity"

        elif membership.role == "agent":
            # No person present: ADR-0007's standing-grant branch (#42 owns its own ASGI tests;
            # kept here so the mechanism is whole for any writing tool, including this one).
            grant = await StandingGrantRepository().get_active(
                session, rc, agent_membership_id=membership.id, tool_name=tool_name
            )
            if grant is None:
                await ApprovalAuditRepository().record(
                    session,
                    rc,
                    kind=audit_kinds.DENIED_FOR_LACK_OF_GRANT,
                    tool_name=tool_name,
                    actor_membership_id=membership.id,
                )
                denial = f"no standing grant authorizes this agent identity to call {tool_name!r}"
            else:
                ctx.deps.pending_approval = ApprovalContext(
                    actor_membership_id=membership.id,
                    tool_name=tool_name,
                    standing_grant_id=grant.id,
                )

        elif not ctx.tool_call_approved:
            settings = get_settings()
            pending = await PendingActionRepository().create(
                session,
                rc,
                conversation_id=conversation_id,
                tool_name=tool_name,
                arguments=arguments,
                tool_call_id=tool_call_id,
                asking_membership_id=membership.id,
                expires_in=timedelta(seconds=settings.pending_action_expiry_seconds),
            )
            await ApprovalAuditRepository().record(
                session,
                rc,
                kind=audit_kinds.REQUESTED,
                tool_name=tool_name,
                actor_membership_id=membership.id,
                pending_action_id=pending.id,
            )
            defer = True

        else:
            pending = await PendingActionRepository().get_by_tool_call(
                session, rc, conversation_id=conversation_id, tool_call_id=tool_call_id
            )
            if pending is None:
                denial = "no pending action found for this tool call"
            else:
                verification = await PendingActionRepository().verify(
                    session,
                    rc,
                    pending_action_id=pending.id,
                    tool_name=tool_name,
                    arguments=arguments,
                )
                if not verification.ok:
                    kind = (
                        audit_kinds.EXPIRED
                        if verification.reason == "expired"
                        else audit_kinds.FAILED_TO_EXECUTE
                    )
                    await ApprovalAuditRepository().record(
                        session,
                        rc,
                        kind=kind,
                        tool_name=tool_name,
                        actor_membership_id=membership.id,
                        pending_action_id=pending.id,
                        details={"reason": verification.reason},
                    )
                    denial = f"approval could not be verified: {verification.reason}"
                elif membership.role not in EXECUTION_ROLES:
                    # The execution-time role re-check ADR-0007 calls for, distinct from -- and run
                    # in addition to -- whatever role the asking membership carried when the write
                    # was first proposed: a membership downgraded in between is refused here.
                    await ApprovalAuditRepository().record(
                        session,
                        rc,
                        kind=audit_kinds.FAILED_TO_EXECUTE,
                        tool_name=tool_name,
                        actor_membership_id=membership.id,
                        pending_action_id=pending.id,
                        details={"reason": "role_not_permitted", "role": membership.role},
                    )
                    denial = f"role {membership.role!r} is no longer permitted to write"
                else:
                    ctx.deps.pending_approval = ApprovalContext(
                        actor_membership_id=membership.id,
                        tool_name=tool_name,
                        pending_action_id=pending.id,
                    )

    # `session` has committed by now (a clean exit from the block above) -- only now does this
    # function raise, so the pending action/audit record it just wrote is already durable before
    # `ApprovalRequired` can reach the event stream encoding the deferred-approval chunk the client
    # sees (#40's ordering acceptance criterion).
    if denial is not None:
        raise ApprovalDenied(denial)
    if defer:
        raise ApprovalRequired()


async def record_write_outcome(
    rc: RequestContext, approval: ApprovalContext | None, *, success: bool
) -> None:
    """Called by a writing tool's own body, after it knows whether the actual write succeeded --
    `executed`/`failed_to_execute` are the tool's own outcome (see
    `app/repositories/approval_audit.py`'s module docstring), never something `require_approval()`
    itself claims to know in advance. A no-op when `approval` is None (the validator never ran, or
    denied the call before recording an `ApprovalContext` -- there is nothing for this call to
    add)."""
    if approval is None:
        return
    async with tenant_session(rc) as session:
        await ApprovalAuditRepository().record(
            session,
            rc,
            kind=audit_kinds.EXECUTED if success else audit_kinds.FAILED_TO_EXECUTE,
            tool_name=approval.tool_name,
            actor_membership_id=approval.actor_membership_id,
            pending_action_id=approval.pending_action_id,
            standing_grant_id=approval.standing_grant_id,
        )


async def resolve_incoming_decisions(
    rc: RequestContext, *, conversation_id: str, decisions: Mapping[str, Any]
) -> None:
    """Resolves every pending action a resumed chat request's approve/refuse decisions name,
    recording the ADR-0007 `approved`/`refused` audit milestone for each. Called by
    `app/api/chat.py` before the agent run itself -- see the module docstring for why a refusal
    can only ever be recorded here, never from inside the tool.

    A decision naming no pending action in this tenant and conversation (an unknown or foreign
    `tool_call_id`, or one already resolved) is silently ignored: there is nothing for an attacker
    to gain from that silence, since the tool's own execution-time `verify()` is the actual gate,
    never this resolution step."""
    if not decisions:
        return
    async with tenant_session(rc) as session:
        membership = await MembershipRepository().get_by_identity(
            session, rc, identity_id=rc.identity_id
        )
        if membership is None:
            return
        for tool_call_id, decision in decisions.items():
            pending = await PendingActionRepository().get_by_tool_call(
                session, rc, conversation_id=conversation_id, tool_call_id=tool_call_id
            )
            if pending is None:
                continue
            approved = decision is True
            resolved = await PendingActionRepository().resolve(
                session,
                rc,
                pending_action_id=pending.id,
                approved=approved,
                resolved_by=membership.id,
            )
            if not resolved:
                continue
            await ApprovalAuditRepository().record(
                session,
                rc,
                kind=audit_kinds.APPROVED if approved else audit_kinds.REFUSED,
                tool_name=pending.tool_name,
                actor_membership_id=membership.id,
                pending_action_id=pending.id,
            )


__all__ = [
    "EXECUTION_ROLES",
    "ApprovalContext",
    "ApprovalDenied",
    "record_write_outcome",
    "require_approval",
    "resolve_incoming_decisions",
]
