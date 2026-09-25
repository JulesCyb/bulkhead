"""MCP server exposing the same tool functions as the backend (app/tools/*).

Payoff: Claude Code / Claude Desktop during development, managed platforms later — without
rewriting the tools.

Context: in production, the tenant/identity context comes from the MCP connection's
authentication (OAuth/token). For local development, from MCP_TENANT_ID / MCP_IDENTITY_ID.

Start (stdio, e.g. in Claude Code's .mcp.json):
    uv run python -m app.mcp.server

Freshness note: MCP Python SDK 2.x -> `from mcp.server.mcpserver import MCPServer`
(previously `from mcp.server.fastmcp import FastMCP`). Check on SDK updates.
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

from mcp.server.mcpserver import MCPServer

from app.config import get_settings
from app.context import RequestContext
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


if __name__ == "__main__":
    server.run(transport="stdio")
