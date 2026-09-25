"""Membership repository (ADR-0003, Spec 2 / #23): the tenant-scoped roster of who belongs to a
tenant, replacing the retired `users` table.

The session comes from `tenant_session(ctx)` and is therefore tenant-bound; RLS filters both
methods to `ctx.tenant_id` before any code here runs. `list_for_tenant` applies no additional
role filtering of its own -- a tenant admin's visibility into a support membership rests on that
property, not on an allow-list of roles this repository would otherwise have to keep in sync.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.context import RequestContext
from app.db.models import Membership


class MembershipRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    identity_id: UUID
    role: str
    created_at: datetime


class MembershipRepository:
    async def get_role(
        self, session: AsyncSession, ctx: RequestContext, *, identity_id: UUID
    ) -> str | None:
        """The role of `identity_id`'s membership in `ctx.tenant_id`, or None -- never
        raises -- when no such membership exists."""
        return (
            await session.execute(
                select(Membership.role).where(
                    Membership.tenant_id == ctx.tenant_id,
                    Membership.identity_id == identity_id,
                )
            )
        ).scalar_one_or_none()

    async def get_by_identity(
        self, session: AsyncSession, ctx: RequestContext, *, identity_id: UUID
    ) -> MembershipRecord | None:
        """The full membership row (id and role) of `identity_id` in `ctx.tenant_id`, or None --
        never raises -- when no such membership exists. Unlike `get_role`, this also returns the
        membership's own id, which the approval mechanism (ADR-0007, `app/tools/approvals.py`)
        needs as `asking_membership_id`/`actor_membership_id` -- a tenant-scoped fact about the
        membership, not the global identity."""
        row = (
            await session.execute(
                select(Membership).where(
                    Membership.tenant_id == ctx.tenant_id,
                    Membership.identity_id == identity_id,
                )
            )
        ).scalar_one_or_none()
        return MembershipRecord.model_validate(row) if row is not None else None

    async def list_for_tenant(
        self, session: AsyncSession, ctx: RequestContext
    ) -> list[MembershipRecord]:
        """Every membership of `ctx.tenant_id`, including support-role ones -- no role-based
        filtering, by construction: the WHERE clause below names only `tenant_id`."""
        rows = (
            await session.execute(select(Membership).where(Membership.tenant_id == ctx.tenant_id))
        ).scalars()
        return [MembershipRecord.model_validate(row) for row in rows]
