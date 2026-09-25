"""The suspension check shared by every place a request's tenant context is resolved, other than
`tenant_session()` itself (Spec 9 / #69, ADR-0010): `app/deps.py`'s AUTH_MODE=dev-headers branch
(AUTH_MODE=jwt already checks inline -- #24), the MCP server's context provider
(`app/mcp/server.py`), and the agent-run entry points (`app/agents/assistant.py`,
`app/api/chat.py`) -- each calls `ensure_tenant_not_suspended(ctx)` once, right after a
`RequestContext` is built and before running any tool or touching any tenant data, independently
of whatever `tenant_session()` will separately re-check the moment a repository actually opens a
session (`app/db/session.py`). Redundant by design: an agent run, or the MCP server, may never
call a tool at all for a given turn, so relying solely on the lazy `tenant_session()` check would
let a suspended tenant get a model response before anything ever touched a session.

Uses the same narrow, cross-tenant-free control-plane read the AUTH_MODE=jwt path already makes
(`control.tenant_auth_settings()` via `TenantAuthSettingsRepository`, over `control_session()`,
which sets no `app.tenant_id` at all) -- never a second mechanism. A tenant with no control-plane
row at all is not suspended (ADR-0002's pooled default).
"""

from __future__ import annotations

from app.context import RequestContext
from app.db.session import TenantSuspendedError, control_session
from app.repositories.control import TenantAuthSettingsRepository

__all__ = ["TenantSuspendedError", "ensure_tenant_not_suspended"]


async def ensure_tenant_not_suspended(ctx: RequestContext) -> None:
    """Raises `TenantSuspendedError` if `ctx.tenant_id` is currently suspended."""
    async with control_session() as session:
        auth_settings = await TenantAuthSettingsRepository().get(session, tenant_id=ctx.tenant_id)
    if auth_settings is not None and auth_settings.suspended:
        raise TenantSuspendedError(ctx.tenant_id)
