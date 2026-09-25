"""Spec 8 / #61: chat and document search actually run against a tenant's *resolved* residency,
not just resolvable in principle -- the live path the assistant and search take.

The lower-level resolution logic (`app.residency.resolve_residency_route`,
`app.llm.resolve_tenant_chat_model`, `app.embeddings.resolve_tenant_embedding_client`) is already
covered end to end against a real, RLS-scoped database in `tests/test_residency.py`,
`tests/test_tenant_model_resolution.py`, and `tests/test_rls_integration.py`. This module covers
the remaining, distinct claim: that the *running* assistant (`/agents/assistant/run`,
`/agents/assistant/stream`, `/api/chat`) and the document-search tool actually call those
resolvers and use whatever they return -- for two tenants of different residencies, and for a
tenant with no resolvable residency -- rather than reaching the removed, deployment-wide
`app.llm.get_model()` or `app.embeddings.embed()` defaults.

No real database, no real model/provider call: `resolve_chat_model`
(`app.agents.assistant`/`app.api.chat`) and `resolve_tenant_embedding_client`
(`app.tools.documents`) are patched per tenant id, mirroring `tests/conftest.py`'s
`resolve_to_model` seam and `tests/test_residency.py`'s fake-session style.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

import httpx
import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.agents import assistant as assistant_module
from app.api import chat as chat_module
from app.context import RequestContext
from app.gateway_credentials import GatewayCredentialUnavailable
from app.llm import ModelNotAllowedForResidency
from app.main import app
from app.residency import ResidencyUnresolved
from app.tools import documents as document_tools


def _headers(identity_id: uuid.UUID | None = None) -> dict[str, str]:
    return {"X-Identity-Id": str(identity_id or uuid.uuid4())}


def _submit_message_body(text: str = "hi") -> dict:
    return {
        "id": "conv-1",
        "trigger": "submit-message",
        "messages": [{"id": "m1", "role": "user", "parts": [{"type": "text", "text": text}]}],
    }


def _model_answering(tag: str) -> FunctionModel:
    """A `FunctionModel` whose only answer names `tag` -- stands in for "the model resolved for
    this tenant's residency route", distinguishable in the response body. Carries a
    `stream_function` too: `/api/chat` always drives its model through `run_stream`."""

    async def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=f"answered via {tag}")])

    async def _respond_stream(messages: list[ModelMessage], info: AgentInfo):
        yield f"answered via {tag}"

    return FunctionModel(_respond, stream_function=_respond_stream)


@pytest.fixture
def asgi_client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# --- AC1: a chat request is answered using the model route of the tenant's own residency ---


async def test_run_endpoint_uses_the_model_resolved_for_each_tenants_residency(
    asgi_client, monkeypatch
):
    """Two tenants, two residencies, two distinct model routes -- the one-shot `/run` endpoint
    actually answers through whichever model `resolve_chat_model` resolved for that request's own
    tenant, never a shared deployment-wide default."""
    tenant_eu, tenant_us = uuid.uuid4(), uuid.uuid4()
    routes = {tenant_eu: _model_answering("eu-route"), tenant_us: _model_answering("us-route")}

    async def _fake_resolve(deps):
        return routes[deps.ctx.tenant_id]

    monkeypatch.setattr(assistant_module, "resolve_chat_model", _fake_resolve)

    async with asgi_client:
        response_eu = await asgi_client.post(
            f"/v1/t/{tenant_eu}/agents/assistant/run",
            json={"prompt": "hi"},
            headers=_headers(),
        )
        response_us = await asgi_client.post(
            f"/v1/t/{tenant_us}/agents/assistant/run",
            json={"prompt": "hi"},
            headers=_headers(),
        )

    assert response_eu.status_code == 200, response_eu.text
    assert response_us.status_code == 200, response_us.text
    assert response_eu.json()["output"] == "answered via eu-route"
    assert response_us.json()["output"] == "answered via us-route"


async def test_chat_endpoint_uses_the_model_resolved_for_each_tenants_residency(
    asgi_client, monkeypatch
):
    """The same claim, driven through `/api/chat` (the Vercel AI SDK adapter) instead of the
    one-shot endpoint."""
    tenant_eu, tenant_us = uuid.uuid4(), uuid.uuid4()
    routes = {tenant_eu: _model_answering("eu-route"), tenant_us: _model_answering("us-route")}

    async def _fake_resolve(deps):
        return routes[deps.ctx.tenant_id]

    async def _fake_load_history(ctx, conversation_id):
        return []

    monkeypatch.setattr(chat_module, "resolve_chat_model", _fake_resolve)
    monkeypatch.setattr(
        assistant_module.conversation_tools, "load_conversation_history", _fake_load_history
    )

    async with asgi_client:
        response_eu = await asgi_client.post(
            f"/v1/t/{tenant_eu}/api/chat", json=_submit_message_body(), headers=_headers()
        )
        response_us = await asgi_client.post(
            f"/v1/t/{tenant_us}/api/chat", json=_submit_message_body(), headers=_headers()
        )

    assert response_eu.status_code == 200, response_eu.text
    assert response_us.status_code == 200, response_us.text
    assert "eu-route" in response_eu.text
    assert "us-route" in response_us.text
    assert "us-route" not in response_eu.text
    assert "eu-route" not in response_us.text


# --- AC3: a tenant with no resolvable residency is refused cleanly, on both chat entry points ---


@pytest.mark.parametrize(
    "raised",
    [
        ResidencyUnresolved("tenant has no usable residency"),
        ModelNotAllowedForResidency("claude", "eu"),
        GatewayCredentialUnavailable("no gateway credential recorded"),
    ],
)
async def test_run_gets_a_clear_failure_not_a_default_route(asgi_client, monkeypatch, raised):
    async def _fake_resolve(deps):
        raise raised

    monkeypatch.setattr(assistant_module, "resolve_chat_model", _fake_resolve)

    async with asgi_client:
        response = await asgi_client.post(
            f"/v1/t/{uuid.uuid4()}/agents/assistant/run",
            json={"prompt": "hi"},
            headers=_headers(),
        )

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error"] == "content_routing_unavailable"


async def test_stream_gets_a_clear_failure_as_a_mapped_sse_error_not_a_default_route(
    asgi_client, monkeypatch
):
    """Model resolution happens before the stream opens, but by the time this generator runs the
    ASGI response has already started (status 200) -- so the failure must surface as a mapped
    `event: error`, not a raw exception on an already-started stream (see `app/api/agents.py`)."""

    async def _fake_resolve(deps):
        raise ResidencyUnresolved("tenant has no usable residency")

    monkeypatch.setattr(assistant_module, "resolve_chat_model", _fake_resolve)

    async with asgi_client:
        response = await asgi_client.post(
            f"/v1/t/{uuid.uuid4()}/agents/assistant/stream",
            json={"prompt": "hi"},
            headers=_headers(),
        )

    assert response.status_code == 200
    assert "event: error" in response.text
    assert "content_routing_unavailable" in response.text


async def test_chat_gets_a_clear_failure_not_a_default_route(asgi_client, monkeypatch):
    async def _fake_resolve(deps):
        raise ResidencyUnresolved("tenant has no usable residency")

    monkeypatch.setattr(chat_module, "resolve_chat_model", _fake_resolve)

    async with asgi_client:
        response = await asgi_client.post(
            f"/v1/t/{uuid.uuid4()}/api/chat", json=_submit_message_body(), headers=_headers()
        )

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error"] == "content_routing_unavailable"


# --- AC2 + AC4: document search embeds against the endpoint resolved for the tenant's own ------
# --- residency; no hard-coded default embedding endpoint is reachable from this path. ----------


class _FakeEmbeddingResponseData:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, embedding: list[float]) -> None:
        self.data = [_FakeEmbeddingResponseData(embedding)]


class _FakeEmbeddingsAPI:
    def __init__(self, embedding: list[float]) -> None:
        self._embedding = embedding
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeEmbeddingResponse(self._embedding)


class _FakeEmbeddingClient:
    """Stands in for the `AsyncOpenAI` `resolve_tenant_embedding_client` would otherwise build --
    tagged with a residency-distinguishable embedding vector, never a real network call."""

    def __init__(self, embedding: list[float]) -> None:
        self.embeddings = _FakeEmbeddingsAPI(embedding)


async def test_search_documents_embeds_against_the_route_resolved_for_the_tenants_residency(
    monkeypatch,
):
    """Two tenants, two residencies, two distinct embedding routes -- `search_documents` embeds
    its query against whichever client `resolve_tenant_embedding_client` resolved for that
    request's own tenant, and the resulting embedding is what actually reaches the repository
    search call (never a process-wide default embedding endpoint -- #61 removed `embed()`)."""
    tenant_eu, tenant_us = uuid.uuid4(), uuid.uuid4()
    clients = {
        tenant_eu: _FakeEmbeddingClient([1.0, 0.0]),
        tenant_us: _FakeEmbeddingClient([0.0, 1.0]),
    }

    async def _fake_resolve(session, ctx, *, settings=None):
        return clients[ctx.tenant_id]

    captured: list[tuple[uuid.UUID, list[float]]] = []

    class _FakeDocumentRepository:
        async def search(self, session, embedding, *, limit):
            captured.append((session, embedding))
            return []

    @asynccontextmanager
    async def _fake_tenant_session(ctx):
        yield ctx.tenant_id  # stands in for a session; only used as an opaque token here

    monkeypatch.setattr(document_tools, "resolve_tenant_embedding_client", _fake_resolve)
    monkeypatch.setattr(document_tools, "DocumentRepository", _FakeDocumentRepository)
    monkeypatch.setattr(document_tools, "tenant_session", _fake_tenant_session)

    ctx_eu = RequestContext(tenant_id=tenant_eu, identity_id=uuid.uuid4())
    ctx_us = RequestContext(tenant_id=tenant_us, identity_id=uuid.uuid4())

    await document_tools.search_documents(ctx_eu, "acme contract")
    await document_tools.search_documents(ctx_us, "acme contract")

    assert clients[tenant_eu].embeddings.calls  # the eu client's own embeddings.create() ran
    assert clients[tenant_us].embeddings.calls
    embeddings_seen = [embedding for _session, embedding in captured]
    assert [1.0, 0.0] in embeddings_seen
    assert [0.0, 1.0] in embeddings_seen


async def test_search_documents_fails_closed_when_residency_is_unresolvable(monkeypatch):
    """#61 AC3: the search path fails exactly like the chat path for a tenant with no resolvable
    residency -- never a fallback to a default embedding endpoint."""

    async def _fake_resolve(session, ctx, *, settings=None):
        raise ResidencyUnresolved(f"tenant {ctx.tenant_id} has no usable residency")

    @asynccontextmanager
    async def _fake_tenant_session(ctx):
        yield object()

    monkeypatch.setattr(document_tools, "resolve_tenant_embedding_client", _fake_resolve)
    monkeypatch.setattr(document_tools, "tenant_session", _fake_tenant_session)

    ctx = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())
    with pytest.raises(ResidencyUnresolved):
        await document_tools.search_documents(ctx, "acme contract")


# --- AC6: the embeddings module docstring states the one-family / re-embedding-migration rule --


def test_embeddings_docstring_states_one_family_shared_across_residencies():
    import app.embeddings as embeddings_module

    doc = " ".join((embeddings_module.__doc__ or "").lower().split())
    assert "one embedding model family" in doc
    assert "residency" in doc and "serving region" in doc
    assert "re-embedding migration" in doc
    assert "new vector column" in doc
