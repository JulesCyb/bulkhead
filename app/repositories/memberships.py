"""Membership repository (ADR-0003, Spec 2 / #23): the tenant-scoped roster of who belongs to a
tenant, replacing the retired `users` table.

The session comes from `tenant_session(ctx)` and is therefore tenant-bound; RLS filters both
methods to `ctx.tenant_id` before any code here runs. `list_for_tenant` applies no additional
role filtering of its own -- a tenant admin's visibility into a support membership rests on that
property, not on an allow-list of roles this repository would otherwise have to keep in sync.

`ensure_membership` is the owner-role side (code review 2026-09-26): the one function the
operator's `create` command writes a tenant's first admin membership through, on its own
owner-role connection -- pooled and dedicated alike -- so no operator module issues SQL against
`memberships` itself (CLAUDE.md rule 3).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict
from sqlalchemy import insert, select, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from app.context import RequestContext, Role
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


async def ensure_membership(
    conn: AsyncConnection, *, tenant_id: UUID, identity_id: UUID, role: Role
) -> Literal["created", "already exists"]:
    """Owner-role write (code review 2026-09-26): gives `identity_id` a membership of `role` in
    `tenant_id` unless it already has one, on the caller's own open owner-role connection.
    Idempotent: an existing membership is reported as `"already exists"` and left as it is, its
    role included.

    **Requires the caller to have set the tenant context on `conn` first.** `memberships` carries
    `FORCE ROW LEVEL SECURITY`, which binds the owner role exactly as it binds `app`: without
    `app.tenant_id` set to `tenant_id` for this transaction the existence check would see nothing
    and the insert would fail its `WITH CHECK`. On the pooled path the control repository's own
    forced-RLS helper has already set it (`ControlRepository.create_tenant_record`/`get_record`,
    one of which `app.operator.create.create_tenant` always calls first on the same transaction);
    on the dedicated path `app.operator.dedicated_db.ensure_dedicated_admin_membership` sets it
    itself against the dedicated database, where the control repository has no connection. This
    function checks that precondition and raises `RuntimeError` when the connection's tenant
    context names a different tenant or none, rather than answering for a tenant it cannot see."""
    current = (
        await conn.execute(text("SELECT current_setting('app.tenant_id', true)"))
    ).scalar_one()
    if current != str(tenant_id):
        raise RuntimeError(
            f"ensure_membership for tenant {tenant_id} needs that tenant context set on the "
            f"connection first; it is {current or 'unset'!r}"
        )
    existing = (
        await conn.execute(
            select(Membership.id).where(
                Membership.tenant_id == tenant_id, Membership.identity_id == identity_id
            )
        )
    ).first()
    if existing is not None:
        return "already exists"
    await conn.execute(
        insert(Membership).values(tenant_id=tenant_id, identity_id=identity_id, role=role)
    )
    return "created"
