"""Tool functions for standing grants (ADR-0005, ADR-0007, Spec 5 / #38) -- a shared module for
agent tools AND the MCP server, following the role-gated admin pattern `app/tools/memberships.py`
establishes (ADR-0004, Spec 3 / #26): a caller's role is checked *first*, before any data access,
with `ctx.require_role("admin")`. A failed check raises `PermissionError`, turned into a 403 by
the application's registered exception handler (`app.main.handle_permission_error`) -- never a
silent allow, never a crash.

`create_standing_grant` and `revoke_standing_grant` are the only way a tenant admin manages who an
agent identity may act as without a person present; `list_standing_grants` is the one place to see
every grant a tenant has ever made, active and revoked alike, exactly as ADR-0007 calls for.
"""

from __future__ import annotations

from uuid import UUID

from app.context import RequestContext
from app.db.session import tenant_session
from app.repositories.standing_grants import StandingGrantRecord, StandingGrantRepository


async def create_standing_grant(
    ctx: RequestContext, *, agent_membership_id: UUID, tool_name: str, granted_by: UUID
) -> StandingGrantRecord:
    """Grant `agent_membership_id` standing permission to call `tool_name` with no person
    present. Admin-only; refused (`NotAnAgentMembership`, from the repository) unless the target
    membership currently carries the `agent` role."""
    ctx.require_role("admin")
    async with tenant_session(ctx) as session:
        grant = await StandingGrantRepository().create(
            session,
            ctx,
            agent_membership_id=agent_membership_id,
            tool_name=tool_name,
            granted_by=granted_by,
        )
        return StandingGrantRecord.model_validate(grant)


async def revoke_standing_grant(ctx: RequestContext, *, grant_id: UUID, revoked_by: UUID) -> bool:
    """Revoke a standing grant -- recorded on the same row, never deleted. Admin-only."""
    ctx.require_role("admin")
    async with tenant_session(ctx) as session:
        return await StandingGrantRepository().revoke(
            session, ctx, grant_id=grant_id, revoked_by=revoked_by
        )


async def list_standing_grants(ctx: RequestContext) -> list[StandingGrantRecord]:
    """List every standing grant the tenant has ever made, active and revoked alike. Admin-only."""
    ctx.require_role("admin")
    async with tenant_session(ctx) as session:
        return await StandingGrantRepository().list_for_tenant(session, ctx)
