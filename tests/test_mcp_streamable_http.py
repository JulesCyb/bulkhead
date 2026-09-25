"""ASGI-seam tests for the MCP server's networked transport (#49, ADR-0005, ADR-0012).

Seam: `httpx` + `ASGITransport` against `app.main.create_app()` with `mcp_transport=
"streamable-http"` -- the same pattern `tests/test_jwt_auth.py` uses for the HTTP path, applied to
the MCP mount instead. The control-plane and membership repositories are faked (no real Postgres),
exactly like `tests/test_jwt_auth.py`; the real signature/tenant-check/RLS chain has its own
coverage in `tests/test_agent_identity_end_to_end_integration.py`.

`MCPServer.streamable_http_app()` (the MCP SDK's own session-negotiation layer -- SSE framing,
protocol handshakes) is swapped for a tiny fake ASGI app that reports back whatever per-connection
`RequestContext` `MCPTenantAuthMiddleware` set before calling it. This keeps the test's seam
exactly at what issue #49 asks for -- the connection's tenant/identity/means resolution -- without
re-implementing the MCP wire protocol; the tool-invocation path itself (a healthy call, a denied
call, a masked exception) already has its own dedicated seam in `tests/test_mcp_tool_errors.py`.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import jwt
import pytest

import app.deps as deps_module
import app.mcp.server as mcp_server
import app.token_verifier as token_verifier_module
from app.config import Settings, get_settings
from app.context import RequestContext

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


@asynccontextmanager
async def _fake_session():
    yield None


def _install_fake_control_plane(monkeypatch, *, auth_settings, identities, memberships):
    """Mirrors `tests/test_jwt_auth.py`'s own helper -- installed on `app.token_verifier` (the
    module `app.deps.get_context` and `app.mcp.server.MCPTenantAuthMiddleware` both call into,
    issue #44)."""

    class FakeTenantAuthSettingsRepository:
        async def get(self, session, *, tenant_id, default_issuer=None):
            if tenant_id not in auth_settings:
                return None
            issuer, suspended = auth_settings[tenant_id]
            return SimpleNamespace(issuer=issuer or default_issuer, suspended=suspended)

    class FakeIdentityRepository:
        async def find_by_issuer_and_subject(self, session, *, issuer, subject):
            identity_id = identities.get((issuer, subject))
            if identity_id is None:
                return None
            return SimpleNamespace(id=identity_id, issuer=issuer, subject=subject)

    class FakeMembershipRepository:
        async def get_role(self, session, ctx: RequestContext, *, identity_id):
            return memberships.get((ctx.tenant_id, identity_id))

    monkeypatch.setattr(token_verifier_module, "control_session", _fake_session)
    monkeypatch.setattr(token_verifier_module, "tenant_session", lambda ctx: _fake_session())
    monkeypatch.setattr(
        token_verifier_module, "TenantAuthSettingsRepository", FakeTenantAuthSettingsRepository
    )
    monkeypatch.setattr(token_verifier_module, "IdentityRepository", FakeIdentityRepository)
    monkeypatch.setattr(token_verifier_module, "MembershipRepository", FakeMembershipRepository)


async def _fake_inner_app(scope, receive, send):
    """Stands in for `MCPServer.streamable_http_app()`'s own session-negotiation app (see module
    docstring): reports back whatever `MCPTenantAuthMiddleware` set as the per-connection
    context, so the test can assert on it without speaking the MCP wire protocol."""
    ctx = mcp_server._connection_context.get()
    body = json.dumps(
        {
            "identity_id": str(ctx.identity_id),
            "roles": sorted(ctx.roles),
            "means_kind": ctx.means.kind if ctx.means else None,
            "means_id": ctx.means.id if ctx.means else None,
        }
    ).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": body})


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
    )
    fields.update(overrides)
    return Settings(**fields)


@pytest.fixture
def mcp_app(monkeypatch):
    """A fresh `create_app()` with the networked MCP transport mounted and its inner
    session-negotiation app swapped for `_fake_inner_app` (see module docstring)."""
    from app import main as main_module

    settings = _mcp_settings()
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    monkeypatch.setattr(mcp_server.server, "streamable_http_app", lambda **kwargs: _fake_inner_app)
    app = main_module.create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    return app, settings


async def _mcp_request(app, tenant_id: uuid.UUID, token: str | None) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(f"/v1/t/{tenant_id}/mcp/", headers=headers)


# --- AC1: a connection whose token names a different tenant than its address is refused ---


async def test_token_naming_a_different_tenant_is_refused(monkeypatch, mcp_app):
    app, _ = mcp_app
    tenant_id = uuid.uuid4()
    other_tenant = uuid.uuid4()
    identity_id = uuid.uuid4()
    _install_fake_control_plane(
        monkeypatch,
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


# --- AC2: a person's token resolves to delegation (actor=person, means=the assistant's tools) ---


async def test_persons_token_resolves_to_delegation(monkeypatch, mcp_app):
    app, _ = mcp_app
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    _install_fake_control_plane(
        monkeypatch,
        auth_settings={tenant_id: (HUMAN_ISSUER, False)},
        identities={(HUMAN_ISSUER, "sub-1"): identity_id},
        memberships={(tenant_id, identity_id): "member"},
    )
    token = _make_token(secret=HUMAN_SECRET, issuer=HUMAN_ISSUER, audience=str(tenant_id))

    response = await _mcp_request(app, tenant_id, token)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["identity_id"] == str(identity_id)
    assert body["means_kind"] == "agent"
    assert body["means_id"] == "assistant"


# --- AC3: an agent identity's token resolves to autonomous use (actor=identity, means=cred) ---


async def test_agent_identity_token_resolves_to_autonomous_use(monkeypatch, mcp_app):
    app, _ = mcp_app
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    _install_fake_control_plane(
        monkeypatch,
        auth_settings={tenant_id: (HUMAN_ISSUER, False)},
        identities={(mcp_server.AGENT_IDENTITY_ISSUER, "agent-sub-1"): identity_id},
        memberships={(tenant_id, identity_id): "agent"},
    )
    token = _make_token(
        secret=AGENT_SECRET,
        issuer=mcp_server.AGENT_IDENTITY_ISSUER,
        subject="agent-sub-1",
        audience=str(tenant_id),
        extra={"cred": "agt_abc123"},
    )

    response = await _mcp_request(app, tenant_id, token)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["identity_id"] == str(identity_id)
    assert body["roles"] == ["agent"]
    assert body["means_kind"] == "credential"
    assert body["means_id"] == "agt_abc123"
