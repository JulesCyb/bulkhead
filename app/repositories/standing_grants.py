"""Standing-grant repository (ADR-0005, ADR-0007, Spec 5 / #38): the only path to
`standing_grants`.

The session comes from `tenant_session(ctx)` and is therefore tenant-bound; RLS filters every
method to `ctx.tenant_id` before any code here runs -- a grant created under one tenant is simply
not a row a second tenant's session can see, by construction.

**Two different rules, enforced in two different places.** Whether the *caller* asking to create
or revoke a grant is a tenant admin is `ctx.require_role("admin")`, checked by the tool
(`app/tools/standing_grants.py`) before this repository is ever reached -- the same role check
Spec 3 (S3-T1 / #26) established. Whether the *target* membership actually carries the `agent`
role, by contrast, is this repository's own invariant: `create()` re-reads that membership's
current role from the database and refuses (`NotAnAgentMembership`) if it is anything else,
including a role that used to be `agent` and has since changed. Neither check substitutes for the
other.

**At most one active grant** per tenant, agent membership, and tool is enforced by the database
itself (`standing_grants_active_uidx`, migration 0031: a partial unique index on
`(agent_membership_id, tool_name) WHERE revoked_at IS NULL`), not by a check-then-insert here that
a race could defeat. `create()` lets the resulting `IntegrityError` propagate rather than
swallowing it -- a caller that wants a friendlier error translates it, but never re-derives the
uniqueness rule in application code.

`revoke()` is the one mutation this table allows after creation: it stamps `revoked_at` and
`revoked_by` on the very row that was granted, never deletes it, so `list_for_tenant` always shows
every grant a tenant has ever made -- active and revoked alike -- with who granted (and, if
applicable, who revoked) each one.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.context import RequestContext
from app.db.models import Membership, StandingGrant
from app.repositories.errors import NotFoundInTenant


class NotAnAgentMembership(NotFoundInTenant):
    """Raised by `create()` when the target membership does not currently carry the `agent`
    role -- including a membership this tenant has no record of at all (RLS makes an unknown id
    and a cross-tenant id indistinguishable, so both land here). Mapped to a 404 by
    `app.main.handle_not_found_in_tenant` (`app.repositories.errors.NotFoundInTenant`'s one
    shared handler) -- previously unmapped, so it fell through to the generic 500 handler
    (finding from the 2026-09-25 review)."""

    public_message = "No such agent membership in this tenant."


class StandingGrantRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    agent_membership_id: UUID
    tool_name: str
    granted_by: UUID
    created_at: datetime
    revoked_at: datetime | None
    revoked_by: UUID | None


class StandingGrantRepository:
    async def create(
        self,
        session: AsyncSession,
        ctx: RequestContext,
        *,
        agent_membership_id: UUID,
        tool_name: str,
        granted_by: UUID,
    ) -> StandingGrant:
        """Grant `agent_membership_id` permission to call `tool_name` with no person present.
        Refuses outright (`NotAnAgentMembership`) unless that membership's role, read fresh right
        here, is exactly `agent`. A second active grant for the same tenant, agent membership,
        and tool raises `sqlalchemy.exc.IntegrityError` from the database's own partial unique
        index -- this method never pre-checks for it, so there is nothing here for a race to
        defeat."""
        role = (
            await session.execute(
                select(Membership.role).where(
                    Membership.tenant_id == ctx.tenant_id,
                    Membership.id == agent_membership_id,
                )
            )
        ).scalar_one_or_none()
        if role != "agent":
            raise NotAnAgentMembership(
                f"membership {agent_membership_id} does not carry the agent role"
            )

        grant = StandingGrant(
            tenant_id=ctx.tenant_id,
            agent_membership_id=agent_membership_id,
            tool_name=tool_name,
            granted_by=granted_by,
        )
        session.add(grant)
        await session.flush()
        return grant

    async def revoke(
        self, session: AsyncSession, ctx: RequestContext, *, grant_id: UUID, revoked_by: UUID
    ) -> bool:
        """Marks an active grant revoked, once, recording who revoked it and when -- never
        deletes the row. Returns False (a no-op) for an unknown id, a cross-tenant id, or a grant
        that is already revoked."""
        result = await session.execute(
            update(StandingGrant)
            .where(
                StandingGrant.tenant_id == ctx.tenant_id,
                StandingGrant.id == grant_id,
                StandingGrant.revoked_at.is_(None),
            )
            .values(revoked_at=func.now(), revoked_by=revoked_by)
        )
        return result.rowcount > 0

    async def get_active(
        self,
        session: AsyncSession,
        ctx: RequestContext,
        *,
        agent_membership_id: UUID,
        tool_name: str,
    ) -> StandingGrant | None:
        """The active (never revoked) grant, if any, authorizing `agent_membership_id` to call
        `tool_name` right now -- the check a writing tool makes before running for an agent
        identity with no person present (ADR-0007). None the instant a matching grant is
        revoked: this always queries live state, never a value cached earlier in the request."""
        return (
            await session.execute(
                select(StandingGrant).where(
                    StandingGrant.tenant_id == ctx.tenant_id,
                    StandingGrant.agent_membership_id == agent_membership_id,
                    StandingGrant.tool_name == tool_name,
                    StandingGrant.revoked_at.is_(None),
                )
            )
        ).scalar_one_or_none()

    async def list_for_tenant(
        self, session: AsyncSession, ctx: RequestContext
    ) -> list[StandingGrantRecord]:
        """Every grant ever made in `ctx.tenant_id`, active and revoked alike -- each naming the
        granting membership and creation time, and the revoking membership and timestamp once
        revoked."""
        rows = (
            await session.execute(
                select(StandingGrant).where(StandingGrant.tenant_id == ctx.tenant_id)
            )
        ).scalars()
        return [StandingGrantRecord.model_validate(row) for row in rows]
