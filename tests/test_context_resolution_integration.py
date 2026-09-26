"""End-to-end proof, per adapter, that `app.context_resolution` is the one chain every entry
point resolves a request's context through (#103, spec #91 "A1"), against real PostgreSQL +
pgvector (`pgserver`, via `uv sync --group dbtest`) -- the real repository-backed
`ControlPlaneReads` adapter, a real signed token, and a real tool call, rather than the
suite-wide fake (`tests/conftest.py`'s `FakeControlPlaneReads`) every unit-level ASGI-seam test
in `tests/test_context_resolution.py`/`tests/test_mcp_streamable_http.py` uses.

Seeding: `tests.support`'s shared `environment`/`seed_tenant` fixtures (issue #96 / spec #90
"A6"), the same pattern `tests/test_tenant_record_integration.py` and
`tests/test_agent_identity_end_to_end_integration.py` already use for a real cluster.

Wire-protocol helpers are reused, not duplicated, from `tests/test_mcp_streamable_http.py`
(`_initialize_session`, `_call_search_documents`, `_running_app`, `_mcp_request`, `_fixed_hit`,
`_MCP_ACCEPT`) and the writing-tool approval round-trip is reused from
`tests/test_writing_tool_approval_integration.py` (`_rename_model`, `_propose_body`,
`_resume_body`, `_headers`, `_chat_path`). Deliberately *not* relocated into `tests/support/` (as
the ticket allows, "if that avoids duplication"): `tests/support/__init__.py` imports `pgserver`
unconditionally, and `tests/test_mcp_streamable_http.py`'s whole point is that most of it runs
with no real database at all -- moving these helpers there would make importing them, and
therefore collecting that file, require `pgserver` too. A plain module-to-module import (this
file already needs `pgserver` for its own fixtures) avoids that regression. Seeding a
conversation directly (the one seeding step below that isn't just `seed_tenant`) goes straight
through `tests.support.seed_conversation`, exactly like the file it borrows the rest of the
round-trip from.

**#117's caveat, which applies to every test below that touches an audit table:**
`approval_audit_events` (the only audit table `app/repositories/approval_audit.py` writes)
records the *actor* membership (ADR-0007's sense: who asked, approved, or was denied) but not the
*means* (ADR-0005's sense: delegation vs. an agent identity's own credential) -- that gap is
tracked, not fixed, by #117. So "the audit row's actor and means" (#103's acceptance criterion) is
provable here only in two separate pieces: the *means* is asserted directly on the
`RequestContext` a tool actually received (`ctx.means`) and, for the HTTP adapter, was already
proven on every span's attributes by `tests/test_context_resolution.py`'s
`test_every_span_of_an_http_run_records_the_person_as_actor_and_the_agent_as_means` -- exactly
what #101 did; the *actor* is proven here on a real `approval_audit_events` row.
"""

from __future__ import annotations

import time
import uuid

import httpx
import jwt
import pytest
from sqlalchemy import text

pgserver = pytest.importorskip("pgserver")

import app.mcp.server as mcp_server  # noqa: E402
from app import main as main_module  # noqa: E402
from app.agents import assistant as assistant_module  # noqa: E402
from app.api import chat as chat_module  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402
from app.context import Means  # noqa: E402
from app.context_resolution import FORBIDDEN_DETAIL  # noqa: E402
from app.db.session import tenant_session  # noqa: E402
from app.main import app as http_app  # noqa: E402
from app.operator.suspend import set_tenant_suspended  # noqa: E402
from app.repositories.agent_credentials import AgentCredentialRepository  # noqa: E402
from app.repositories.agent_identities import AgentIdentityRepository  # noqa: E402
from app.token_verifier import set_default_adapter_for_tests  # noqa: E402
from tests.conftest import resolve_to_model  # noqa: E402
from tests.support import (  # noqa: E402
    SeededTenant,
    cluster,
    environment,
    seed_conversation,
    seed_tenant,
)
from tests.test_mcp_streamable_http import (  # noqa: E402
    _MCP_ACCEPT,
    _call_search_documents,
    _fixed_hit,
    _initialize_session,
    _mcp_request,
    _running_app,
)
from tests.test_writing_tool_approval_integration import (  # noqa: E402
    CONVERSATION_ID,
    NEW_TITLE,
    TOOL_CALL_ID,
    _chat_path,
    _headers,
    _propose_body,
    _rename_model,
    _resume_body,
)

_ = (cluster, environment)

SECRET = "context-resolution-integration-human-secret-32b"
AGENT_SECRET = "context-resolution-integration-agent-secret-32b"
ALLOWED_HOST = "mcp.example.com"


@pytest.fixture(autouse=True)
def _real_control_plane_reads():
    """Every test in this file drives a real cluster; the suite-wide fake
    (`tests/conftest.py`'s autouse `not_suspended`) would answer every read from memory and
    nothing would reach the database at all -- exactly the gap this file exists to close."""
    set_default_adapter_for_tests(None)
    yield


def _jwt_settings(**overrides) -> Settings:
    fields = dict(
        _env_file=None,
        environment="test",
        auth_mode="jwt",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        default_identity_issuer="seed",
        jwt_verification_key=SECRET,
        jwt_algorithm="HS256",
        agent_token_signing_key=AGENT_SECRET,
        agent_token_ttl_seconds=300,
        mcp_transport="stdio",
    )
    fields.update(overrides)
    return Settings(**fields)


def _person_token(tenant: SeededTenant, *, role: str = "member", ttl: float = 300.0) -> str:
    """A real, signed token for one of `seed_tenant`'s seeded identities -- every identity
    `tests.support.seeding.seed_tenant` writes carries issuer `'seed'`, its own id as subject."""
    now = int(time.time())
    claims = {
        "iss": "seed",
        "sub": str(tenant.identities[role]),
        "aud": str(tenant.tenant_id),
        "iat": now,
        "exp": now + ttl,
    }
    return jwt.encode(claims, SECRET, algorithm="HS256")


@pytest.fixture
def mcp_settings() -> Settings:
    return _jwt_settings(mcp_transport="streamable-http", mcp_allowed_hosts=ALLOWED_HOST)


@pytest.fixture
def mcp_app(monkeypatch, mcp_settings):
    """A fresh `create_app()` with the real, networked MCP transport mounted (real
    `MCPTenantAuthMiddleware`, real `MCPServer.streamable_http_app()`) -- unlike
    `tests/test_mcp_streamable_http.py`'s own `mcp_app` fixture, `run_role_rls_guard` is left
    untouched: this file has a real, migrated, correctly-roled cluster (`environment`), so the
    guard that fails closed against a privileged connection or missing RLS is exercised for real,
    not skipped."""
    monkeypatch.setattr(main_module, "get_settings", lambda: mcp_settings)
    app = main_module.create_app()
    app.dependency_overrides[get_settings] = lambda: mcp_settings
    return app


async def _http_run(tenant_id: uuid.UUID, token: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=http_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            f"/v1/t/{tenant_id}/agents/assistant/run",
            json={"prompt": "Find the contract"},
            headers={"Authorization": f"Bearer {token}"},
        )


# --- (a) HTTP person: a real bearer token, a real tool call, over the one-shot run route --------


async def test_http_person_reaches_the_tool_as_delegation_with_a_tenant_record(
    environment, monkeypatch, test_model
):
    """AC (a): the tool receives a context carrying the member's own identity, role `member`,
    delegation as the means (`("agent", "assistant")`), and a `tenant_record` -- resolved by
    `app.context_resolution.resolve_bearer_context` against the real control plane, through
    `app.deps.get_context`."""
    tenant = await seed_tenant(environment, residency="eu", roles=["member"])
    settings = _jwt_settings()
    http_app.dependency_overrides[get_settings] = lambda: settings

    captured: dict = {}

    async def fake_search(ctx, query, limit=5):
        captured["ctx"] = ctx
        return [_fixed_hit()]

    monkeypatch.setattr(assistant_module.document_tools, "search_documents", fake_search)

    try:
        response = await _http_run(tenant.tenant_id, _person_token(tenant, role="member"))
    finally:
        http_app.dependency_overrides.pop(get_settings, None)

    assert response.status_code == 200, response.text
    ctx = captured["ctx"]
    assert ctx.tenant_id == tenant.tenant_id
    assert ctx.identity_id == tenant.identities["member"]
    assert ctx.roles == frozenset({"member"})
    assert ctx.means == Means(kind="agent", id="assistant")
    assert ctx.tenant_record is not None
    assert ctx.tenant_record.tenant_id == tenant.tenant_id


# --- (b) MCP person: the same token, the real mount, the real wire protocol ----------------------


async def test_mcp_person_reaches_the_tool_as_delegation_with_a_tenant_record(
    environment, mcp_app, monkeypatch
):
    """AC (b): identical facts to (a), but resolved by `MCPTenantAuthMiddleware` -- itself only an
    adapter of `resolve_bearer_context` (#101/#102) -- and observed through a real `tools/call`
    over the real Streamable HTTP mount, not a fake inner app."""
    tenant = await seed_tenant(environment, residency="eu", roles=["member"])
    token = _person_token(tenant, role="member")

    captured: dict = {}

    async def fake_search(ctx, query, limit=5):
        captured["ctx"] = ctx
        return [_fixed_hit()]

    monkeypatch.setattr(mcp_server.document_tools, "search_documents", fake_search)

    path = f"/v1/t/{tenant.tenant_id}/mcp/"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": _MCP_ACCEPT,
        "Host": ALLOWED_HOST,
    }
    transport = httpx.ASGITransport(app=mcp_app)
    async with (
        _running_app(mcp_app),
        httpx.AsyncClient(transport=transport, base_url="http://localhost") as client,
    ):
        session_headers = await _initialize_session(client, path, headers)
        result = await _call_search_documents(client, path, session_headers)

    assert result["isError"] is False
    ctx = captured["ctx"]
    assert ctx.tenant_id == tenant.tenant_id
    assert ctx.identity_id == tenant.identities["member"]
    assert ctx.has_role("member")
    assert ctx.means == Means(kind="agent", id="assistant")
    assert ctx.tenant_record is not None
    assert ctx.tenant_record.tenant_id == tenant.tenant_id


# --- (c) MCP agent identity: a real credential, exchanged for a real token over the real route --


async def test_mcp_agent_identity_reaches_the_tool_naming_its_credential(
    environment, mcp_app, mcp_settings, monkeypatch
):
    """AC (c): an agent identity's own credential, created and issued the way
    `tests/test_agent_identity_end_to_end_integration.py` proves the repository layer does it,
    exchanged for a real access token through the real `POST /v1/t/{tenant_id}/agent-tokens`
    route (`app/api/agent_tokens.py`, `app/agent_credential_exchange.py`) -- not minted by hand --
    then used over the real MCP mount. The tool receives role `agent` and the credential's own
    public id as the means (`RequestContext.acting_through("credential", ...)`, ADR-0005)."""
    tenant = await seed_tenant(environment, residency="eu", roles=["admin"])
    admin_ctx = tenant.ctx("admin")
    async with tenant_session(admin_ctx) as session:
        created = await AgentIdentityRepository().create(session, admin_ctx, name="nightly sync")
        issued = await AgentCredentialRepository().create(
            session, admin_ctx, identity_id=created.identity_id, name="nightly sync cred"
        )

    http_app.dependency_overrides[get_settings] = lambda: mcp_settings
    try:
        transport = httpx.ASGITransport(app=http_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            token_response = await client.post(
                f"/v1/t/{tenant.tenant_id}/agent-tokens",
                json={"public_id": issued.public_id, "secret": issued.secret},
            )
    finally:
        http_app.dependency_overrides.pop(get_settings, None)
    assert token_response.status_code == 200, token_response.text
    access_token = token_response.json()["access_token"]

    captured: dict = {}

    async def fake_search(ctx, query, limit=5):
        captured["ctx"] = ctx
        return [_fixed_hit()]

    monkeypatch.setattr(mcp_server.document_tools, "search_documents", fake_search)

    path = f"/v1/t/{tenant.tenant_id}/mcp/"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": _MCP_ACCEPT,
        "Host": ALLOWED_HOST,
    }
    transport = httpx.ASGITransport(app=mcp_app)
    async with (
        _running_app(mcp_app),
        httpx.AsyncClient(transport=transport, base_url="http://localhost") as client,
    ):
        session_headers = await _initialize_session(client, path, headers)
        result = await _call_search_documents(client, path, session_headers)

    assert result["isError"] is False
    ctx = captured["ctx"]
    assert ctx.tenant_id == tenant.tenant_id
    assert ctx.identity_id == created.identity_id
    assert ctx.has_role("agent")
    assert ctx.means == Means(kind="credential", id=issued.public_id)


# --- (d) audit row: the actor is real; the means is only ever a context/span fact (#117) ---------


async def test_writing_tool_approval_audit_row_names_the_acting_membership(
    environment, monkeypatch
):
    """AC (d): drives one real writing-tool approval round-trip on `/api/chat`
    (`tests/test_writing_tool_approval_integration.py`'s own machinery -- a cheap, already-proven
    path, reused rather than rebuilt) against a seeded tenant, then reads the resulting
    `approval_audit_events` row back directly: its `actor_membership_id` is the seeded member's
    own membership.

    Per the module docstring's #117 caveat: this table has no `means` column at all today (only
    the *approval* means -- a pending action or standing grant id, neither asked for by this
    ticket's criterion -- and the acting membership). The *delegation* means (ADR-0005:
    `("agent", "assistant")` for this same request) is not persisted anywhere a database query
    could read back; it is instead a fact of the `RequestContext`/trace span, proven for the HTTP
    adapter by `tests/test_context_resolution.py`'s
    `test_every_span_of_an_http_run_records_the_person_as_actor_and_the_agent_as_means`. This test
    only proves the actor half of #103's "actor and means" criterion, on purpose.
    """
    tenant = await seed_tenant(environment, residency="eu", roles=["member"], documents=1)
    identity_id = tenant.identities["member"]
    membership_id = tenant.memberships["member"]
    document_id = tenant.document_ids[0]

    await seed_conversation(
        tenant.cluster,
        tenant_id=tenant.tenant_id,
        identity_id=identity_id,
        conversation_id=CONVERSATION_ID,
    )

    model = _rename_model(document_id=document_id, title=NEW_TITLE)
    monkeypatch.setattr(chat_module, "resolve_chat_model", resolve_to_model(model))

    transport = httpx.ASGITransport(app=http_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        propose = await client.post(
            _chat_path(tenant.tenant_id), json=_propose_body(), headers=_headers(identity_id)
        )
        assert propose.status_code == 200, propose.text
        resumed = await client.post(
            _chat_path(tenant.tenant_id),
            json=_resume_body(
                tool_call_id=TOOL_CALL_ID,
                document_id=document_id,
                title=NEW_TITLE,
                approved=True,
            ),
            headers=_headers(identity_id),
        )
    assert resumed.status_code == 200, resumed.text

    async with tenant.superuser_connection() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT kind, actor_membership_id FROM approval_audit_events "
                    "WHERE tenant_id = :tid ORDER BY seq"
                ),
                {"tid": tenant.tenant_id},
            )
        ).all()

    kinds = [row.kind for row in rows]
    assert "requested" in kinds
    assert "approved" in kinds
    assert "executed" in kinds
    assert all(row.actor_membership_id == membership_id for row in rows)


# --- Also: a suspended seeded tenant is refused on both transports, with the same body -----------


async def test_suspended_tenant_is_refused_identically_on_http_and_mcp(environment, mcp_app):
    tenant = await seed_tenant(environment, residency="eu", roles=["member"])
    token = _person_token(tenant, role="member")

    async with tenant.owner_connection() as conn:
        async with conn.begin():
            result = await set_tenant_suspended(conn, str(tenant.tenant_id), suspended=True)
    assert result.changed

    settings = _jwt_settings()
    http_app.dependency_overrides[get_settings] = lambda: settings
    try:
        http_response = await _http_run(tenant.tenant_id, token)
    finally:
        http_app.dependency_overrides.pop(get_settings, None)

    mcp_response = await _mcp_request(mcp_app, tenant.tenant_id, token)

    assert http_response.status_code == 403
    assert http_response.json() == {"detail": FORBIDDEN_DETAIL}
    assert mcp_response.status_code == 403
    assert mcp_response.json() == {"detail": FORBIDDEN_DETAIL}
