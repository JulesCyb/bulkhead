"""MCP server exposing the same tool functions as the backend (app/tools/*).

Payoff: Claude Code / Claude Desktop during development, managed platforms later — without
rewriting the tools.

Context: in production, the tenant/identity context comes from the MCP connection's
authentication (OAuth/token, `app.token_verifier`), per connection. For local development
(`MCP_TRANSPORT=stdio`, the default), from the process-wide MCP_TENANT_ID / MCP_IDENTITY_ID.
Which transport is active is a single setting (`Settings.mcp_transport`, issue #48 / ADR-0005),
guarded at startup by `check_mcp_mode` below the same way `AUTH_MODE=dev-headers` is guarded by
`app.main.check_auth_mode`.

Every tool resolves its context through `resolve_context()`, not `context_provider()` directly
(Spec 9 / #69, ADR-0010): it builds the context, then checks suspension
(`app.tenant_suspension.ensure_tenant_not_suspended`) before any tool body runs -- the MCP
connection handler's own, independent check, alongside the HTTP API's (`app/deps.py`) and the
agent-run entry points' (`app/agents/assistant.py`).

Start (stdio, e.g. in Claude Code's .mcp.json):
    uv run python -m app.mcp.server

Networked (`MCP_TRANSPORT=streamable-http`, issue #49 / ADR-0005): this module builds the tool
server itself; the connection-authenticated ASGI app is `build_streamable_http_app()`, mounted by
`app.main.create_app()` under the tenant's own path prefix (`/v1/t/{tenant_id}/mcp`, ADR-0012) --
MCP now mounts inside the existing API service rather than adding a new one. Every connection's
tenant/identity is derived from its own bearer token via `MCPTenantAuthMiddleware` below, which
reuses the exact same shared check (`app.token_verifier.verify_tenant_token`) the HTTP API's
`app.deps.get_context` does -- never a second, drifting copy of it.

Freshness note: MCP Python SDK 2.x -> `from mcp.server.mcpserver import MCPServer`
(previously `from mcp.server.fastmcp import FastMCP`). Check on SDK updates.
"""

from __future__ import annotations

import contextvars
import functools
import logging
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any
from uuid import UUID

from mcp.server.mcpserver import MCPServer
from starlette.datastructures import Headers
from starlette.responses import JSONResponse

from app.config import Settings, get_settings
from app.context import RequestContext
from app.deps import get_key_source
from app.startup_checks import run_startup_checks
from app.tenant_suspension import TenantSuspendedError, ensure_tenant_not_suspended
from app.token_verifier import (
    AGENT_IDENTITY_ISSUER,
    TenantTokenVerificationError,
    VerificationFailureReason,
    verify_tenant_token,
)
from app.tools import documents as document_tools
from app.tools import memberships as membership_tools

log = logging.getLogger(__name__)

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

server = MCPServer(
    name="ai-app-tools",
    instructions=(
        "This application's tools: semantic search in the tenant's documents, "
        "and (admin-only) listing the tenant's memberships."
    ),
)


def _context_from_env() -> RequestContext:
    s = get_settings()
    if not (s.mcp_tenant_id and s.mcp_identity_id):
        raise RuntimeError("Set MCP_TENANT_ID and MCP_IDENTITY_ID (development only).")
    return RequestContext(tenant_id=UUID(s.mcp_tenant_id), identity_id=UUID(s.mcp_identity_id))


# Per-connection context for the networked transport (issue #49): a `contextvars.ContextVar`
# rather than a mutable module-level value, so concurrent connections (each its own asyncio task
# under Streamable HTTP) never see each other's tenant/identity -- the same isolation a per-request
# FastAPI dependency gets for free, reproduced here since MCP tool functions take no request
# object of their own to thread a context through.
_connection_context: contextvars.ContextVar[RequestContext | None] = contextvars.ContextVar(
    "mcp_connection_context", default=None
)


def _context_from_connection() -> RequestContext:
    ctx = _connection_context.get()
    if ctx is None:  # pragma: no cover - defensive; every request path sets it first
        raise RuntimeError(
            "No per-connection MCP context is set -- MCPTenantAuthMiddleware must authenticate "
            "a connection before any tool call runs on it."
        )
    return ctx


# The seam for production: replaced with `_context_from_connection` (per-connection, streamable
# HTTP) or left as `_context_from_env` (stdio, development). Anything but the env fallback MUST be
# per-connection — a process-wide identity on a shared transport would leak tenants.
context_provider: Callable[[], RequestContext] = _context_from_env


async def resolve_context() -> RequestContext:
    """The MCP connection handler's own context resolution: builds the context, then rejects a
    suspended tenant before any tool body runs (Spec 9 / #69, ADR-0010) -- see module docstring.
    Every tool calls this, never `context_provider()` directly.
    """
    ctx = context_provider()
    await ensure_tenant_not_suspended(ctx.tenant_id)
    return ctx


# Every non-`PermissionError` exception a tool raises comes back as exactly this text -- never the
# exception's own message (issue #49's acceptance criterion 4, matching how
# `app.main.handle_unhandled_exception` never leaks a raw exception to an HTTP caller). The real
# error is logged server-side, not lost -- see `_masked` below.
GENERIC_TOOL_ERROR = "This tool call failed. The error has been logged."


def _masked(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Wraps a tool function so any exception but `PermissionError`/`TenantSuspendedError` is
    logged server-side and replaced with a single generic message before it ever reaches
    `MCPServer._handle_call_tool`'s own catch-all (which otherwise answers with the exception's
    own `str(e)` -- see that method's source). `PermissionError` is deliberately let through
    unmasked: issue #27 already relies on its exact message (the missing role) reaching the
    caller, the same way the HTTP API's 403 body names it. `TenantSuspendedError` (issue #69) is
    the same kind of controlled, expected rejection -- `resolve_context()` above already raises it
    before this wrapper's own function body ever runs, and every other place a context is resolved
    (`app/deps.py`, `app/api/chat.py`, `app/api/agents.py`) lets it propagate as itself rather than
    folding it into a generic 500/masked error."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await fn(*args, **kwargs)
        except (PermissionError, TenantSuspendedError):
            raise
        except Exception:
            log.exception("MCP tool call raised an exception -- masked from the client")
            raise RuntimeError(GENERIC_TOOL_ERROR) from None

    return wrapper


@server.tool()
@_masked
async def search_documents(query: str, limit: int = 5) -> list[dict]:
    """Semantic search in the current tenant's documents."""
    ctx = await resolve_context()
    hits = await document_tools.search_documents(ctx, query, limit)
    return [hit.model_dump(mode="json") for hit in hits]


@server.tool()
@_masked
async def list_memberships() -> list[dict]:
    """List the current tenant's memberships (identity, role, joined-at). Admin-only.

    #27: a caller without the `admin` role raises `PermissionError` from
    `app.tools.memberships.list_memberships` -> `RequestContext.require_role`. `_masked` above
    re-raises `PermissionError` unchanged; it is never caught here. The MCP SDK's own
    tool-invocation dispatch (`MCPServer._handle_call_tool`, the single path every transport calls
    to run any tool) then catches it and answers with a structured `CallToolResult(is_error=True)`
    naming the missing role, instead of letting the exception cross the invocation boundary and
    take the connection down -- the same wrapper every other tool call goes through, not a special
    case added here for this one tool.
    """
    records = await membership_tools.list_memberships(await resolve_context())
    return [record.model_dump(mode="json") for record in records]


# --- Networked transport: per-connection context from a verified bearer token (issue #49) ---

# Every rejected connection gets exactly this body, mirroring `app.deps.FORBIDDEN_DETAIL` -- the
# specific reason is discoverable only from the server-side log, never the response (ADR-0012).
_MCP_FORBIDDEN_DETAIL = "Not authorized for this tenant."


def _actor_context(tenant_id: UUID, resolved) -> RequestContext:  # noqa: ANN001
    """Builds the per-connection `RequestContext` from a verified token, naming the means
    (ADR-0005, issue #43): a person's token resolves to delegation (means = the assistant's
    tools); an agent identity's token resolves to autonomous use (means = the credential that
    authenticated it, from the token's own `cred` claim -- see `app/token_verifier.py`)."""
    base_ctx = RequestContext(
        tenant_id=tenant_id, identity_id=resolved.identity_id, roles=frozenset({resolved.role})
    )
    if resolved.issuer == AGENT_IDENTITY_ISSUER:
        return base_ctx.acting_through("credential", resolved.credential_public_id or "unknown")
    return base_ctx.acting_through("agent", "assistant")


class MCPTenantAuthMiddleware:
    """Wraps the MCP Streamable-HTTP ASGI app with the same three-way tenant check ADR-0012
    requires of the HTTP API: the connection's bearer token is verified via the exact module the
    HTTP path uses (`app.token_verifier.verify_tenant_token`, issue #44), and its resolved
    identity/role become a per-connection `RequestContext` (via `_connection_context` above) for
    the lifetime of that one ASGI request -- never the process-wide `_context_from_env` fallback.

    Mounted under `/v1/t/{tenant_id}/mcp` (`app.main.create_app`), so `tenant_id` arrives as an
    ordinary Starlette path parameter, exactly like every other tenant-scoped route -- the tenant
    the connection is trying to reach, checked against the token's own audience by
    `verify_tenant_token` itself (ADR-0012's three-way check, reused rather than reinvented).
    """

    def __init__(self, app: ASGIApp, *, settings: Settings) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        tenant_id_raw = scope.get("path_params", {}).get("tenant_id")
        try:
            tenant_id = UUID(str(tenant_id_raw))
        except (TypeError, ValueError):
            await JSONResponse({"detail": "Not found."}, status_code=404)(scope, receive, send)
            return

        headers = Headers(scope=scope)
        authorization = headers.get("authorization")
        if not authorization or not authorization.lower().startswith("bearer "):
            response = JSONResponse(
                {"detail": "Missing or malformed bearer token"}, status_code=401
            )
            await response(scope, receive, send)
            return
        token = authorization.split(" ", 1)[1].strip()
        if not token:
            response = JSONResponse(
                {"detail": "Missing or malformed bearer token"}, status_code=401
            )
            await response(scope, receive, send)
            return

        key_source = get_key_source(self.settings)
        try:
            resolved = await verify_tenant_token(
                token,
                tenant_id=tenant_id,
                key_source=key_source,
                default_issuer=self.settings.default_identity_issuer,
                algorithms=(self.settings.jwt_algorithm,),
            )
        except TenantTokenVerificationError as exc:
            if exc.reason is VerificationFailureReason.INVALID_OR_EXPIRED:
                detail = (
                    "No token issuer configured"
                    if exc.issuer is None
                    else "Invalid or expired token"
                )
                response = JSONResponse({"detail": detail}, status_code=401)
            else:
                log.warning(
                    "MCP connection rejected",
                    extra={
                        "event": "mcp_auth_forbidden",
                        "reason": exc.reason.value,
                        "tenant_id": str(tenant_id),
                        "issuer": exc.issuer or "",
                    },
                )
                response = JSONResponse({"detail": _MCP_FORBIDDEN_DETAIL}, status_code=403)
            await response(scope, receive, send)
            return

        ctx = _actor_context(tenant_id, resolved)
        reset_token = _connection_context.set(ctx)
        try:
            await self.app(scope, receive, send)
        finally:
            _connection_context.reset(reset_token)


def build_streamable_http_app(settings: Settings) -> ASGIApp:
    """The networked transport's ASGI app (issue #49): the MCP SDK's own Streamable HTTP app,
    wrapped with `MCPTenantAuthMiddleware` above. `app.main.create_app` mounts this under
    `/v1/t/{tenant_id}/mcp` only when `settings.mcp_transport == "streamable-http"` -- the stdio
    entrypoint (`main()` below) never touches this function, so local development is unaffected.
    """
    inner = server.streamable_http_app(streamable_http_path="/")
    return MCPTenantAuthMiddleware(inner, settings=settings)


def check_mcp_mode(settings: Settings) -> None:
    """Refuse to start with a half-finished MCP transport configuration -- structurally mirrors
    `app.main.check_auth_mode` (issue #48 / ADR-0005): the guardrail lives in code, not only in
    the docs.

    Two independent failure modes:

    - The stdio transport's process-wide identity fallback (`_context_from_env`, above) is only
      reachable when `mcp_transport` is `stdio`. Exactly like `AUTH_MODE=dev-headers`, that
      fallback must never be reachable outside `dev`/`test`: a single, unauthenticated identity
      shared by every connection would leak tenants the moment this process is reachable from
      more than one caller.
    - Any other transport (`streamable-http`) is refused unless a token verifier is configured
      (`jwt_verification_key`, the same signing configuration `app.token_verifier` checks
      connections against) -- regardless of environment, so a half-finished deployment can never
      silently serve every tenant's documents to whoever can open a connection.
    """
    if settings.mcp_transport == "stdio":
        if settings.environment not in ("dev", "test"):
            raise RuntimeError(
                "MCP_TRANSPORT=stdio (the default) relies on the process-wide MCP_TENANT_ID/ "
                "MCP_IDENTITY_ID identity fallback, for local development only -- the same "
                "unauthenticated shortcut AUTH_MODE=dev-headers is guarded against. Set "
                "MCP_TRANSPORT=streamable-http (with JWT_VERIFICATION_KEY configured) or set "
                "ENVIRONMENT=dev."
            )
        return
    if settings.jwt_verification_key is None:
        raise RuntimeError(
            "MCP_TRANSPORT=streamable-http requires a configured token verifier "
            "(JWT_VERIFICATION_KEY) so app.token_verifier has something to check connections "
            "against -- without it, a networked transport would accept a tool call from whoever "
            "can open a connection."
        )


def main() -> None:
    """The MCP server's own startup path: the transport guard above (issue #48 / ADR-0005) runs
    first, fail-closed, then the same fail-closed residency/model-allow-list checks the HTTP API
    runs at construction and lifespan (issue #59 / ADR-0008,
    `app.startup_checks.run_startup_checks`) -- one startup-checks function, not a second
    validation this entry point would otherwise have to reconcile by hand -- before this process
    ever accepts a tool call on either transport."""
    settings = get_settings()
    check_mcp_mode(settings)
    run_startup_checks(settings)
    server.run(transport=settings.mcp_transport)


if __name__ == "__main__":
    main()
