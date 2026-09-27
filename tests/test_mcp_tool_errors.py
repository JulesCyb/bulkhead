"""S3-T2 / #27: a role-check failure raised inside a tool called through the MCP server comes
back as a clean, structured tool-level error, never an exception that crosses the invocation
boundary and could take the stdio connection down.

Seam: a direct, in-process call into `MCPServer._handle_call_tool` -- the one tool-invocation
path every transport (stdio, sse, streamable-http) dispatches every tool call through
(`MCPServer.__init__` wires `on_call_tool=self._handle_call_tool` into the low-level server
unconditionally of transport). This is the same "call the real function directly, in-process"
pattern `tests/test_hardening.py` uses for exercising a guard function without a live server, and
`tests/test_memberships_api.py` uses for faking `app.tools.memberships` under the ASGI seam --
applied here to the MCP tool-invocation seam instead.

`_handle_call_tool` requires a `ServerRequestContext` positional argument, but only forwards it
into `Context(request_context=...)`, which our tool functions never request (no tool below takes
a `Context` parameter) -- so `None` is a faithful stand-in for a real per-connection request
context in these tests.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pytest
from mcp_types import CallToolRequestParams

import app.tools.memberships as membership_tools_module
from app.context import RequestContext
from app.mcp import server as mcp_server
from app.repositories.memberships import MembershipRecord


@asynccontextmanager
async def _fake_session():
    yield None


def _install_fake_memberships(monkeypatch, records: list[MembershipRecord]):
    """Fakes `app.tools.memberships.MembershipRepository.list_for_tenant` -- unrelated to (and not
    migrated by) #100's control-plane-reads adapter, which only covers the narrower `get_role`
    read `app.token_verifier` uses; this is the tenant-scoped listing route's own repository."""

    class _FakeMembershipListingRepository:
        async def list_for_tenant(self, session, ctx: RequestContext):
            return records

    monkeypatch.setattr(membership_tools_module, "tenant_session", lambda ctx: _fake_session())
    monkeypatch.setattr(
        membership_tools_module, "MembershipRepository", _FakeMembershipListingRepository
    )


def _context_with_role(role: str) -> RequestContext:
    return RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4(), roles=frozenset({role}))


def _set_connection_context(role: str) -> None:
    """Installs a context on the per-connection contextvar `resolve_context()` reads first
    (issue #89) -- the seam every tool call in this file goes through instead of the deleted
    `context_provider` indirection. No explicit teardown: `contextvars.ContextVar.set` only ever
    mutates the copy of the context `asyncio` gave this test's own task (each `pytest-asyncio`
    test runs in a fresh task), so it never leaks into another test."""
    mcp_server._connection_context.set(_context_with_role(role))


async def _call(name: str, arguments: dict | None = None):
    params = CallToolRequestParams(name=name, arguments=arguments or {})
    # `None`: see module docstring -- no tool below reads the MCP `Context`.
    return await mcp_server.server._handle_call_tool(None, params)


async def test_denied_membership_listing_returns_a_structured_tool_error(monkeypatch):
    """Acceptance criterion 1: a non-admin caller gets a structured tool error, not a raised
    exception past the invocation boundary."""
    _set_connection_context("member")
    _install_fake_memberships(monkeypatch, [])

    result = await _call("list_memberships")

    assert result.is_error is True
    assert "role 'admin' required" in result.content[0].text


async def test_denied_call_names_the_same_missing_role_as_the_http_403(monkeypatch):
    """Acceptance criterion 3: the same missing-role phrasing the HTTP 403 handler
    (`app.main.handle_permission_error`) puts in its body -- `PermissionError`'s own message,
    matched by `app.main._REQUIRED_ROLE_RE`."""
    _set_connection_context("support")
    _install_fake_memberships(monkeypatch, [])

    result = await _call("list_memberships")

    assert result.is_error is True
    assert "role 'admin' required" in result.content[0].text


async def test_a_denied_call_does_not_affect_a_later_unrelated_call(monkeypatch):
    """Acceptance criterion 2: the same invocation path is exercised for every tool, not a
    special case for `list_memberships` -- a denial on one call leaves the server instance
    healthy for a completely unrelated tool call afterward."""
    _set_connection_context("member")
    _install_fake_memberships(monkeypatch, [])

    denied = await _call("list_memberships")
    assert denied.is_error is True

    async def fake_search_documents(ctx, query, limit=5):
        return []

    monkeypatch.setattr(mcp_server.document_tools, "search_documents", fake_search_documents)

    healthy = await _call("search_documents", {"query": "hello"})

    assert healthy.is_error is False


async def test_a_generic_exception_is_masked_and_never_reaches_the_caller(monkeypatch, caplog):
    """Issue #49's acceptance criterion 4: a non-`PermissionError` exception raised inside a tool
    (a database error, a model provider's raw error body, anything) comes back as the one generic
    message (`mcp_server.GENERIC_TOOL_ERROR`), never its own text -- the real error is only in the
    server-side log."""
    _set_connection_context("member")

    async def _boom(ctx, query, limit=5):
        raise RuntimeError('column "secret_column" does not exist -- a raw DB error')

    monkeypatch.setattr(mcp_server.document_tools, "search_documents", _boom)

    with caplog.at_level("ERROR"):
        result = await _call("search_documents", {"query": "hello"})

    assert result.is_error is True
    # The MCP SDK's own tool dispatch (`ToolManager.call_tool`) wraps whatever text a raised
    # exception carries as "Error executing tool <name>: <text>" -- what matters here is that the
    # wrapped text is our generic message, never the original exception's.
    assert mcp_server.GENERIC_TOOL_ERROR in result.content[0].text
    assert "secret_column" not in result.content[0].text
    # The real error is only in the server-side log (via `log.exception`, exc_info + traceback).
    assert "secret_column" in caplog.text


async def test_a_masked_exception_does_not_affect_a_later_unrelated_call(monkeypatch):
    """The wrapper only replaces this one call's outcome -- a completely different, healthy tool
    call on the same server instance afterward is unaffected."""
    _set_connection_context("admin")
    _install_fake_memberships(monkeypatch, [])

    async def _boom(ctx, query, limit=5):
        raise RuntimeError("boom")

    monkeypatch.setattr(mcp_server.document_tools, "search_documents", _boom)
    masked = await _call("search_documents", {"query": "hello"})
    assert masked.is_error is True
    assert mcp_server.GENERIC_TOOL_ERROR in masked.content[0].text

    healthy = await _call("list_memberships")
    assert healthy.is_error is False


async def test_admin_call_still_succeeds_through_the_same_wrapper(monkeypatch):
    """The wrapper only intercepts a raised exception -- it never mangles a healthy result."""
    identity_id = uuid.uuid4()
    _set_connection_context("admin")
    _install_fake_memberships(
        monkeypatch,
        [
            MembershipRecord(
                id=uuid.uuid4(),
                identity_id=identity_id,
                role="admin",
                created_at=datetime.now(UTC),
            )
        ],
    )

    result = await _call("list_memberships")

    assert result.is_error is False
    assert result.structured_content["result"][0]["identity_id"] == str(identity_id)


if __name__ == "__main__":
    pytest.main([__file__])
