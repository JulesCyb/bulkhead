"""MCP server exposing the same tool functions as the backend (app/tools/*).

Payoff: Claude Code / Claude Desktop during development, managed platforms later — without
rewriting the tools.

Context: in production, the tenant/identity context comes from the MCP connection's
authentication (OAuth/token, `app.token_verifier`), per connection. For local development
(`MCP_TRANSPORT=stdio`, the default), from the process-wide MCP_TENANT_ID / MCP_IDENTITY_ID.
Which transport is active is a single setting (`Settings.mcp_transport`, issue #48 / ADR-0005),
guarded at startup by `check_mcp_mode` below the same way `AUTH_MODE=dev-headers` is guarded by
`app.main.check_auth_mode`.

Start (stdio, e.g. in Claude Code's .mcp.json):
    uv run python -m app.mcp.server

Freshness note: MCP Python SDK 2.x -> `from mcp.server.mcpserver import MCPServer`
(previously `from mcp.server.fastmcp import FastMCP`). Check on SDK updates.
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

from mcp.server.mcpserver import MCPServer

from app.config import Settings, get_settings
from app.context import RequestContext
from app.startup_checks import run_startup_checks
from app.tools import documents as document_tools

server = MCPServer(
    name="ai-app-tools",
    instructions="This application's tools: semantic search in the tenant's documents.",
)


def _context_from_env() -> RequestContext:
    s = get_settings()
    if not (s.mcp_tenant_id and s.mcp_identity_id):
        raise RuntimeError("Set MCP_TENANT_ID and MCP_IDENTITY_ID (development only).")
    return RequestContext(tenant_id=UUID(s.mcp_tenant_id), identity_id=UUID(s.mcp_identity_id))


# The seam for production: replace this with a function that derives tenant/identity from the
# MCP connection's authentication (OAuth/token). Anything but the env fallback MUST be
# per-connection — a process-wide identity on a shared transport would leak tenants.
context_provider: Callable[[], RequestContext] = _context_from_env


@server.tool()
async def search_documents(query: str, limit: int = 5) -> list[dict]:
    """Semantic search in the current tenant's documents."""
    hits = await document_tools.search_documents(context_provider(), query, limit)
    return [hit.model_dump(mode="json") for hit in hits]


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
