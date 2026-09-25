"""Tool functions for memberships — a shared module for agent tools AND the MCP server.

`list_memberships` is the role-gated worked example ADR-0004 and Spec 3 (S3-T1 / #26) call for:
a caller's role is checked *first*, before any data access, with `ctx.require_role("admin")`. A
failed check raises `PermissionError`, which the application's registered exception handler
(`app.main.handle_permission_error`) turns into a 403 — never a 500, and never silently allowed.

Per CLAUDE.md: a role check belongs here (a tool) and in the route that calls it, never inside a
repository's read path — `MembershipRepository.list_for_tenant` applies no role filtering of its
own, by design (see its own docstring), so this is the one and only gate.
"""

from __future__ import annotations

from app.context import RequestContext
from app.db.session import tenant_session
from app.repositories.memberships import MembershipRecord, MembershipRepository


async def list_memberships(ctx: RequestContext) -> list[MembershipRecord]:
    """List the calling tenant's memberships (identity, role, joined-at). Admin-only."""
    ctx.require_role("admin")
    async with tenant_session(ctx) as session:
        return await MembershipRepository().list_for_tenant(session, ctx)
