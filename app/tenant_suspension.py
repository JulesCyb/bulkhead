"""The suspension check shared by every place a request's tenant context is resolved, other than
`tenant_session()` itself (Spec 9 / #69, ADR-0010): `app/deps.py`'s AUTH_MODE=dev-headers branch
*and* its AUTH_MODE=jwt branch (the latter checks after `app/token_verifier.py::verify_tenant_token`
succeeds but before a membership is looked up, since suspension isn't part of that shared token
check -- issue #44), the MCP server's context provider (`app/mcp/server.py`), and the agent-run
entry points (`app/agents/assistant.py`, `app/api/chat.py`) -- each calls
`ensure_tenant_not_suspended(tenant_id)` once, right after `tenant_id` is known and before running
any tool or touching any tenant data, independently of whatever `tenant_session()` will separately
re-check the moment a repository actually opens a session (`app/db/session.py`). Redundant by
design: an agent run, or the MCP server, may never call a tool at all for a given turn, so relying
solely on the lazy `tenant_session()` check would let a suspended tenant get a model response
before anything ever touched a session.

Takes a bare `tenant_id`, not a `RequestContext`: the AUTH_MODE=jwt branch has a tenant id (from
the URL path) before it has resolved an identity or a role, so a full context isn't always
available yet at the point suspension needs checking.

Uses the same narrow, cross-tenant-free control-plane read the AUTH_MODE=jwt path already made
before #44's refactor (`control.tenant_auth_settings()` via `TenantAuthSettingsRepository`, over
`control_session()`, which sets no `app.tenant_id` at all) -- never a second mechanism. A tenant
with no control-plane row at all is not suspended (ADR-0002's pooled default).
"""

from __future__ import annotations

from uuid import UUID

from app.db.session import TenantSuspendedError, control_session
from app.repositories.control import TenantAuthSettingsRepository

__all__ = ["TenantSuspendedError", "ensure_tenant_not_suspended"]


async def ensure_tenant_not_suspended(tenant_id: UUID) -> None:
    """Raises `TenantSuspendedError` if `tenant_id` is currently suspended."""
    async with control_session() as session:
        auth_settings = await TenantAuthSettingsRepository().get(session, tenant_id=tenant_id)
    if auth_settings is not None and auth_settings.suspended:
        raise TenantSuspendedError(tenant_id)
