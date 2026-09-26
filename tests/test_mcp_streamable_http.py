"""ASGI-seam tests for the MCP server's networked transport (#49, ADR-0005, ADR-0012).

Seam: `httpx` + `ASGITransport` against `app.main.create_app()` with `mcp_transport=
"streamable-http"` -- the same pattern `tests/test_jwt_auth.py` uses for the HTTP path, applied to
the MCP mount instead. The control-plane and membership repositories are faked (no real Postgres),
exactly like `tests/test_jwt_auth.py`; the real signature/tenant-check/RLS chain has its own
coverage in `tests/test_agent_identity_end_to_end_integration.py`.

The two rejection tests (AC1 below) never reach `MCPServer.streamable_http_app()`'s own
session-negotiation layer at all -- `MCPTenantAuthMiddleware` answers a rejected connection itself,
before calling the wrapped app -- so they run against the real mount directly, no fake needed.
Every other test speaks the actual MCP Streamable HTTP wire protocol -- JSON-RPC `initialize`,
`notifications/initialized`, then `tools/call` -- over the real mount, through the real
`MCPServer.streamable_http_app()`, with `app.tools.documents.search_documents` faked only at the
tool-function boundary to capture the `RequestContext` it was actually called with. The
tool-invocation path itself (a healthy call, a denied call, a masked exception) has its own
dedicated seam in `tests/test_mcp_tool_errors.py`.

(#102 retired the fake ASGI app this module used to swap in for `MCPServer.streamable_http_app()`,
which reported back whatever `MCPTenantAuthMiddleware` had set on `_connection_context` directly.
Making the middleware a thin adapter of `app.context_resolution.resolve_bearer_context` left
nothing left to prove by bypassing it that the real-wire-protocol tests below don't already prove
through an actual tool call.)

Issue #89 closed the gap between this module's own seam and `tests/test_mcp_tool_errors.py`'s:
neither used to exercise `MCPTenantAuthMiddleware`'s write and a tool's `resolve_context()` read
together. No monkeypatching of `context_provider` (it no longer exists) or `_connection_context`.

Issue #116 removed two workarounds the real-wire-protocol tests used to need: entering
`MCPServer.session_manager` themselves (`app.main.lifespan` now does that, for any
`streamable-http` deployment, not just a test) and forcing a `Host: localhost` header (the mount
now configures a real `MCP_ALLOWED_HOSTS` allow-list, exercised here instead of bypassed).
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import jwt
import pytest

import app.context_resolution as context_resolution_module
import app.deps as deps_module
import app.mcp.server as mcp_server
from app.config import Settings, get_settings
from app.context_resolution import ContextRejection, RejectionReason, RejectionStatus
from app.repositories.documents import DocumentHit
from app.token_verifier import AGENT_IDENTITY_ISSUER, set_default_adapter_for_tests
from tests.conftest import FakeControlPlaneReads

HUMAN_SECRET = "mcp-streamable-http-test-human-secret-32-bytes"
AGENT_SECRET = "mcp-streamable-http-test-agent-secret-32-bytes!"
HUMAN_ISSUER = "https://idp.example.com"


def _make_token(
    *,
    secret: str,
    issuer: str,
    subject: str = "sub-1",
    audience: str | None = None,
    exp_delta: float = 300.0,
    algorithm: str = "HS256",
    extra: dict | None = None,
) -> str:
    now = int(time.time())
    claims = {
        "iss": issuer,
        "sub": subject,
        "aud": audience,
        "iat": now,
        "exp": now + exp_delta,
        **(extra or {}),
    }
    return jwt.encode(claims, secret, algorithm=algorithm)


def _install_fake_control_plane(*, auth_settings, identities, memberships):
    """Mirrors `tests/test_jwt_auth.py`'s own helper: installs one `FakeControlPlaneReads` (#100)
    as the default adapter both `app.deps.get_context` and `app.mcp.server.MCPTenantAuthMiddleware`
    fall back to (both call into `app.token_verifier.verify_tenant_token`, issue #44)."""
    set_default_adapter_for_tests(
        FakeControlPlaneReads(
            auth_settings=auth_settings, identities=identities, memberships=memberships
        )
    )


MCP_ALLOWED_HOST = "mcp.example.com"


def _mcp_settings(**overrides) -> Settings:
    fields = dict(
        _env_file=None,
        environment="prod",
        auth_mode="jwt",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        default_identity_issuer=None,
        jwt_verification_key=HUMAN_SECRET,
        jwt_algorithm="HS256",
        agent_token_signing_key=AGENT_SECRET,
        agent_token_ttl_seconds=300,
        mcp_transport="streamable-http",
        # Issue #116: `check_mcp_mode` refuses `streamable-http` without this; the real-wire-
        # protocol tests below send exactly this Host header to prove it's honoured, not the MCP
        # SDK's own localhost-only default.
        mcp_allowed_hosts=MCP_ALLOWED_HOST,
    )
    fields.update(overrides)
    return Settings(**fields)


async def _mcp_request(app, tenant_id: uuid.UUID, token: str | None) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(f"/v1/t/{tenant_id}/mcp/", headers=headers)


# --- AC1: a connection whose token names a different tenant than its address is refused ---
#
# Both tests below reject the connection inside `MCPTenantAuthMiddleware` itself, before it ever
# calls the wrapped app -- so the real (not faked) `MCPServer.streamable_http_app()` the `mcp_app`
# fixture below builds is never actually reached, and no wire-protocol handshake or running
# lifespan is needed to observe the rejection.


async def test_token_naming_a_different_tenant_is_refused(mcp_app):
    app, _ = mcp_app
    tenant_id = uuid.uuid4()
    other_tenant = uuid.uuid4()
    identity_id = uuid.uuid4()
    _install_fake_control_plane(
        auth_settings={tenant_id: (HUMAN_ISSUER, False)},
        identities={(HUMAN_ISSUER, "sub-1"): identity_id},
        memberships={(tenant_id, identity_id): "member"},
    )
    token = _make_token(secret=HUMAN_SECRET, issuer=HUMAN_ISSUER, audience=str(other_tenant))

    response = await _mcp_request(app, tenant_id, token)

    assert response.status_code == 403
    assert response.json()["detail"] == deps_module.FORBIDDEN_DETAIL


async def test_missing_token_is_unauthorized(mcp_app):
    app, _ = mcp_app
    response = await _mcp_request(app, uuid.uuid4(), None)
    assert response.status_code == 401


# --- AC2/AC3 (person -> delegation, agent identity -> credential): covered below by the real
# wire-protocol tests, which assert the identical facts (identity, role, means) through an actual
# tool call rather than a fake inner app reporting `_connection_context` back directly. ---


# --- #89: the real wire protocol, real tool dispatch, no fake inner app -----------------------

_MCP_ACCEPT = "application/json, text/event-stream"
# The configured allow-list (`_mcp_settings()` above sets `mcp_allowed_hosts=MCP_ALLOWED_HOST`,
# issue #116) -- proves `build_streamable_http_app` honours a real deployment's own Host header,
# not the MCP SDK's own localhost-only default (`host="127.0.0.1"`).
_ALLOWED_HOST_HEADER = {"Host": MCP_ALLOWED_HOST}
_UNLISTED_HOST_HEADER = {"Host": "evil.example.com"}
_INITIALIZE_PARAMS = {
    "protocolVersion": "2025-06-18",
    "capabilities": {},
    "clientInfo": {"name": "bulkhead-test-client", "version": "0.1"},
}


@pytest.fixture
def mcp_app(monkeypatch):
    """A fresh `create_app()` with the networked MCP transport mounted, its *real*
    `MCPServer.streamable_http_app()` -- nothing here fakes the inner session-negotiation app (the
    fake that used to stand in for it, reading `_connection_context` directly, is retired: #102
    made `MCPTenantAuthMiddleware` a thin adapter, so there is nothing left to prove by bypassing
    it). Every test in this module speaks the actual wire protocol -- or, for the two rejection
    tests above, is rejected before the wire protocol ever starts.

    `run_role_rls_guard` is substituted with a no-op: driving the real ASGI lifespan (issue #116,
    `_running_app` below) now reaches it too, and it needs a real database this unit test has
    none of -- unrelated to what this fixture actually exercises (the MCP mount's own startup and
    transport security)."""
    from app import main as main_module

    async def _noop_role_rls_guard() -> None:
        return None

    settings = _mcp_settings()
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    monkeypatch.setattr(main_module, "run_role_rls_guard", _noop_role_rls_guard)
    app = main_module.create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    return app, settings


@asynccontextmanager
async def _running_app(app):
    """Drives the ASGI application's own lifespan around the enclosed block (issue #116): this is
    what now actually starts `MCPServer.session_manager` -- via `app.main.lifespan`, entered for
    real, not a test-side substitute for it. `httpx.ASGITransport` never sends `lifespan` scope
    messages on its own, so tests drive it explicitly through `app.router.lifespan_context`, the
    same pattern `tests/test_hardening.py` already uses for the auth/RLS guards.

    Used as `async with _running_app(app):` directly inside each test body, never as a
    `pytest.fixture` spanning a `yield` -- the session manager's own cancel scope must exit in the
    same task it was entered in, and a fixture's setup/teardown can run as two separate tasks
    under `pytest-asyncio`."""
    async with app.router.lifespan_context(app):
        yield


async def _rpc_call(
    client: httpx.AsyncClient, path: str, headers: dict, body: dict
) -> tuple[int, str | None, dict | None]:
    """POSTs one JSON-RPC message and returns `(status_code, mcp_session_id, payload)`.

    The MCP SDK's `streamable_http_app()` answers a request (as opposed to a notification) over
    an SSE stream when `json_response` is left at its default `False` -- exactly how
    `app.mcp.server.build_streamable_http_app` calls it -- so this parses the `data:` line of the
    single `message` event the server sends for one request, rather than assuming a plain JSON
    body. A request the transport-security middleware rejects outright (issue #116) never reaches
    that layer at all and answers with a plain-text body instead -- `payload` stays `None` for it,
    never a `JSONDecodeError`."""
    async with client.stream("POST", path, json=body, headers=headers) as response:
        status_code = response.status_code
        session_id = response.headers.get("mcp-session-id")
        content_type = response.headers.get("content-type", "")
        payload: dict | None = None
        if "text/event-stream" in content_type:
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data = line[len("data:") :].strip()
                    if data:
                        payload = json.loads(data)
        elif "application/json" in content_type:
            raw = await response.aread()
            if raw:
                payload = json.loads(raw)
        else:
            await response.aread()
        return status_code, session_id, payload


async def _initialize_session(client: httpx.AsyncClient, path: str, headers: dict) -> dict:
    """Runs the handshake (`initialize` then `notifications/initialized`) and returns the
    headers a subsequent `tools/call` on the same session must send."""
    status_code, session_id, payload = await _rpc_call(
        client,
        path,
        headers,
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": _INITIALIZE_PARAMS},
    )
    assert status_code == 200, payload
    assert session_id, "initialize must hand back an Mcp-Session-Id"
    session_headers = {**headers, "Mcp-Session-Id": session_id}

    status_code, _, _ = await _rpc_call(
        client, path, session_headers, {"jsonrpc": "2.0", "method": "notifications/initialized"}
    )
    assert status_code == 202

    return session_headers


async def _call_search_documents(
    client: httpx.AsyncClient, path: str, headers: dict, *, request_id: int = 2
) -> dict:
    status_code, _, payload = await _rpc_call(
        client,
        path,
        headers,
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": "search_documents", "arguments": {"query": "hello"}},
        },
    )
    assert status_code == 200, payload
    return payload["result"]


def _fixed_hit() -> DocumentHit:
    return DocumentHit(id=uuid.uuid4(), title="Runbook", snippet="A fixed hit.", score=0.9)


async def test_real_wire_protocol_hands_search_documents_the_verified_persons_context(
    monkeypatch, mcp_app
):
    """AC1 (person's token): over the real mount, speaking the real wire protocol, the tool
    receives the context carrying that token's identity and role, and `means.kind == "agent"`."""
    app, _ = mcp_app
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    _install_fake_control_plane(
        auth_settings={tenant_id: (HUMAN_ISSUER, False)},
        identities={(HUMAN_ISSUER, "sub-1"): identity_id},
        memberships={(tenant_id, identity_id): "member"},
    )
    token = _make_token(secret=HUMAN_SECRET, issuer=HUMAN_ISSUER, audience=str(tenant_id))

    captured: dict = {}

    async def fake_search_documents(ctx, query, limit=5):
        captured["ctx"] = ctx
        return [_fixed_hit()]

    monkeypatch.setattr(mcp_server.document_tools, "search_documents", fake_search_documents)

    path = f"/v1/t/{tenant_id}/mcp/"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": _MCP_ACCEPT,
        **_ALLOWED_HOST_HEADER,
    }
    transport = httpx.ASGITransport(app=app)
    async with (
        _running_app(app),
        httpx.AsyncClient(transport=transport, base_url="http://localhost") as client,
    ):
        session_headers = await _initialize_session(client, path, headers)
        result = await _call_search_documents(client, path, session_headers)

    assert result["isError"] is False
    assert "ctx" in captured, "search_documents must have been called"
    ctx = captured["ctx"]
    assert ctx.tenant_id == tenant_id
    assert ctx.identity_id == identity_id
    assert ctx.has_role("member")
    assert ctx.means is not None
    assert ctx.means.kind == "agent"


async def test_real_wire_protocol_hands_search_documents_the_agent_identitys_context(
    monkeypatch, mcp_app
):
    """AC1 (agent identity's token): same wire protocol, `means.kind == "credential"`."""
    app, _ = mcp_app
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    _install_fake_control_plane(
        auth_settings={tenant_id: (HUMAN_ISSUER, False)},
        identities={(AGENT_IDENTITY_ISSUER, "agent-sub-1"): identity_id},
        memberships={(tenant_id, identity_id): "agent"},
    )
    token = _make_token(
        secret=AGENT_SECRET,
        issuer=AGENT_IDENTITY_ISSUER,
        subject="agent-sub-1",
        audience=str(tenant_id),
        extra={"cred": "agt_abc123"},
    )

    captured: dict = {}

    async def fake_search_documents(ctx, query, limit=5):
        captured["ctx"] = ctx
        return [_fixed_hit()]

    monkeypatch.setattr(mcp_server.document_tools, "search_documents", fake_search_documents)

    path = f"/v1/t/{tenant_id}/mcp/"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": _MCP_ACCEPT,
        **_ALLOWED_HOST_HEADER,
    }
    transport = httpx.ASGITransport(app=app)
    async with (
        _running_app(app),
        httpx.AsyncClient(transport=transport, base_url="http://localhost") as client,
    ):
        session_headers = await _initialize_session(client, path, headers)
        result = await _call_search_documents(client, path, session_headers)

    assert result["isError"] is False
    ctx = captured["ctx"]
    assert ctx.tenant_id == tenant_id
    assert ctx.identity_id == identity_id
    assert ctx.means is not None
    assert ctx.means.kind == "credential"
    assert ctx.means.id == "agt_abc123"


async def test_two_concurrent_connections_each_see_only_their_own_tenant(monkeypatch, mcp_app):
    """AC2: two concurrent connections for two different tenants -- each `tools/call` sees only
    its own connection's tenant, never the other's, even though both run through the same shared
    MCP server instance concurrently (`asyncio.gather`)."""
    app, _ = mcp_app
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    identity_a, identity_b = uuid.uuid4(), uuid.uuid4()
    _install_fake_control_plane(
        auth_settings={tenant_a: (HUMAN_ISSUER, False), tenant_b: (HUMAN_ISSUER, False)},
        identities={
            (HUMAN_ISSUER, "sub-a"): identity_a,
            (HUMAN_ISSUER, "sub-b"): identity_b,
        },
        memberships={(tenant_a, identity_a): "member", (tenant_b, identity_b): "member"},
    )
    token_a = _make_token(
        secret=HUMAN_SECRET, issuer=HUMAN_ISSUER, subject="sub-a", audience=str(tenant_a)
    )
    token_b = _make_token(
        secret=HUMAN_SECRET, issuer=HUMAN_ISSUER, subject="sub-b", audience=str(tenant_b)
    )

    seen: list = []

    async def fake_search_documents(ctx, query, limit=5):
        seen.append(ctx)
        return [_fixed_hit()]

    monkeypatch.setattr(mcp_server.document_tools, "search_documents", fake_search_documents)

    async def _run(tenant_id: uuid.UUID, token: str) -> dict:
        path = f"/v1/t/{tenant_id}/mcp/"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": _MCP_ACCEPT,
            **_ALLOWED_HOST_HEADER,
        }
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
            session_headers = await _initialize_session(client, path, headers)
            return await _call_search_documents(client, path, session_headers)

    async with _running_app(app):
        result_a, result_b = await asyncio.gather(_run(tenant_a, token_a), _run(tenant_b, token_b))

    assert result_a["isError"] is False
    assert result_b["isError"] is False
    assert len(seen) == 2
    seen_by_tenant = {ctx.tenant_id: ctx for ctx in seen}
    assert seen_by_tenant[tenant_a].identity_id == identity_a
    assert seen_by_tenant[tenant_b].identity_id == identity_b
    # Never the other connection's tenant on either call.
    assert seen_by_tenant[tenant_a].tenant_id != seen_by_tenant[tenant_b].tenant_id


async def test_real_wire_protocol_rejects_an_unlisted_host_header(mcp_app):
    """Issue #116: `build_streamable_http_app` now passes `MCP_ALLOWED_HOSTS` through as the SDK's
    `TransportSecuritySettings.allowed_hosts` -- a Host header outside that list is rejected by the
    SDK's own DNS-rebinding middleware (421), proving the allow-list is actually enforced and not
    just accepted-and-ignored configuration."""
    app, _ = mcp_app
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    _install_fake_control_plane(
        auth_settings={tenant_id: (HUMAN_ISSUER, False)},
        identities={(HUMAN_ISSUER, "sub-1"): identity_id},
        memberships={(tenant_id, identity_id): "member"},
    )
    token = _make_token(secret=HUMAN_SECRET, issuer=HUMAN_ISSUER, audience=str(tenant_id))

    path = f"/v1/t/{tenant_id}/mcp/"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": _MCP_ACCEPT,
        **_UNLISTED_HOST_HEADER,
    }
    transport = httpx.ASGITransport(app=app)
    async with (
        _running_app(app),
        httpx.AsyncClient(transport=transport, base_url="http://localhost") as client,
    ):
        status_code, _, _ = await _rpc_call(
            client,
            path,
            headers,
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": _INITIALIZE_PARAMS},
        )

    assert status_code == 421


async def test_streamable_http_refuses_a_tools_call_with_no_connection_context(monkeypatch):
    """AC3: under `streamable-http`, a `tools/call` with no per-connection context set is
    refused -- never served from the environment identity. Exercised directly against
    `resolve_context()` (the seam `search_documents`/`list_memberships` both call): no HTTP
    request can reach a tool without `MCPTenantAuthMiddleware` setting the contextvar first, so
    this is the only way to observe "unset" for this transport."""
    settings = _mcp_settings()
    monkeypatch.setattr(mcp_server, "get_settings", lambda: settings)
    assert mcp_server._connection_context.get() is None

    with pytest.raises(RuntimeError, match="per-connection"):
        await mcp_server.resolve_context()


# --- Delegation proof: the MCP adapter renders whatever the module returns, no second copy -----


async def test_mcp_adapter_renders_a_rejection_it_did_not_compute(monkeypatch, mcp_app):
    """The MCP twin of `tests/test_context_resolution.py`'s own delegation proof
    (`test_dev_headers_adapter_renders_a_rejection_it_did_not_compute`): a perfectly valid,
    verifiable token for an unsuspended tenant -- yet the substituted `resolve_bearer_context`
    says the tenant is suspended, and `MCPTenantAuthMiddleware` answers exactly that. Proves the
    middleware only renders what the module returns; it never re-implements a check of its own."""
    app, _ = mcp_app
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    _install_fake_control_plane(
        auth_settings={tenant_id: (HUMAN_ISSUER, False)},
        identities={(HUMAN_ISSUER, "sub-1"): identity_id},
        memberships={(tenant_id, identity_id): "member"},
    )
    token = _make_token(secret=HUMAN_SECRET, issuer=HUMAN_ISSUER, audience=str(tenant_id))
    rejection = ContextRejection(
        status=RejectionStatus.FORBIDDEN,
        reason=RejectionReason.TENANT_SUSPENDED,
        detail=context_resolution_module.FORBIDDEN_DETAIL,
        request_id="req-mcp-delegation",
        issuer=HUMAN_ISSUER,
    )
    seen: list[dict] = []

    async def _fake(**kwargs):
        seen.append(kwargs)
        return rejection

    monkeypatch.setattr(context_resolution_module, "resolve_bearer_context", _fake)

    response = await _mcp_request(app, tenant_id, token)

    assert response.status_code == 403
    assert response.json()["detail"] == context_resolution_module.FORBIDDEN_DETAIL
    assert seen and seen[0]["tenant_id"] == tenant_id


# --- Both adapters render the same ContextRejection identically ---------------------------------


def test_mcp_rejection_matches_the_http_adapters_for_the_same_reason():
    """#102 acceptance criterion: the same `ContextRejection` renders identically on both
    transports. `app.deps._render` (HTTPException, turned into a `{"detail": ...}` body by
    FastAPI's own default exception handler) and `app.mcp.server._render` (JSONResponse,
    constructing that same body directly) are two different response *types* standing in for the
    same status and detail -- never two independently-decided answers for one rejection."""
    tenant_id = uuid.uuid4()
    rejection = ContextRejection(
        status=RejectionStatus.FORBIDDEN,
        reason=RejectionReason.TENANT_SUSPENDED,
        detail=context_resolution_module.FORBIDDEN_DETAIL,
        request_id="req-parity",
        issuer=HUMAN_ISSUER,
    )

    http_exc = deps_module._render(rejection, tenant_id)
    mcp_response = mcp_server._render(rejection, tenant_id)

    assert http_exc.status_code == mcp_response.status_code
    assert json.loads(mcp_response.body)["detail"] == http_exc.detail
