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

No real database, no real model/provider call: the run's model resolver is installed per tenant
id through `tests/conftest.py`'s `route_run` (the run module's own test hook,
`app.agents.run.set_run_collaborators_for_tests`), and `resolve_tenant_embedding_client`
(`app.tools.documents`) is patched per tenant id -- except in the #105 section, where the real
resolver runs from the tenant record the request's context carries
(`FakeControlPlaneReads.records`) and only client construction is replaced, to observe which model
name the run actually used.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

import httpx
import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.agents.run import prepare_run
from app.config import get_settings
from app.context import RequestContext
from app.gateway_credentials import GatewayCredentialUnavailable
from app.llm import ModelNotAllowedForResidency
from app.main import app
from app.residency import ResidencyAllowList, ResidencyUnresolved
from app.tenant_record import TenantRecord
from app.tenant_settings import TenantSettings
from app.token_verifier import set_default_adapter_for_tests
from app.tools import conversations as conversation_tools
from app.tools import documents as document_tools
from tests.conftest import FakeControlPlaneReads


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
    asgi_client, route_run
):
    """Two tenants, two residencies, two distinct model routes -- the one-shot `/run` endpoint
    actually answers through whichever model the run's resolver resolved for that request's own
    tenant record, never a shared deployment-wide default."""
    tenant_eu, tenant_us = uuid.uuid4(), uuid.uuid4()
    routes = {tenant_eu: _model_answering("eu-route"), tenant_us: _model_answering("us-route")}

    def _fake_resolve(record, *, settings=None):
        return routes[record.tenant_id]

    route_run(model_resolver=_fake_resolve)

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
    asgi_client, monkeypatch, route_run
):
    """The same claim, driven through `/api/chat` (the Vercel AI SDK adapter) instead of the
    one-shot endpoint."""
    tenant_eu, tenant_us = uuid.uuid4(), uuid.uuid4()
    routes = {tenant_eu: _model_answering("eu-route"), tenant_us: _model_answering("us-route")}

    def _fake_resolve(record, *, settings=None):
        return routes[record.tenant_id]

    async def _fake_load_history(ctx, conversation_id):
        return []

    route_run(model_resolver=_fake_resolve)
    # The chat route still builds its own tool dependencies until #108.
    monkeypatch.setattr(conversation_tools, "load_conversation_history", _fake_load_history)

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
async def test_run_gets_a_clear_failure_not_a_default_route(asgi_client, route_run, raised):
    def _fake_resolve(record, *, settings=None):
        raise raised

    route_run(model_resolver=_fake_resolve)

    async with asgi_client:
        response = await asgi_client.post(
            f"/v1/t/{uuid.uuid4()}/agents/assistant/run",
            json={"prompt": "hi"},
            headers=_headers(),
        )

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error"] == "content_routing_unavailable"


async def test_stream_gets_a_clear_failure_as_a_mapped_sse_error_not_a_default_route(
    asgi_client, route_run
):
    """Model resolution happens before the stream opens, but by the time this generator runs the
    ASGI response has already started (status 200) -- so the failure must surface as a mapped
    `event: error`, not a raw exception on an already-started stream (see `app/api/agents.py`)."""

    def _fake_resolve(record, *, settings=None):
        raise ResidencyUnresolved("tenant has no usable residency")

    route_run(model_resolver=_fake_resolve)

    async with asgi_client:
        response = await asgi_client.post(
            f"/v1/t/{uuid.uuid4()}/agents/assistant/stream",
            json={"prompt": "hi"},
            headers=_headers(),
        )

    assert response.status_code == 200
    assert "event: error" in response.text
    assert "content_routing_unavailable" in response.text


async def test_chat_gets_a_clear_failure_not_a_default_route(asgi_client, route_run):
    def _fake_resolve(record, *, settings=None):
        raise ResidencyUnresolved("tenant has no usable residency")

    route_run(model_resolver=_fake_resolve)

    async with asgi_client:
        response = await asgi_client.post(
            f"/v1/t/{uuid.uuid4()}/api/chat", json=_submit_message_body(), headers=_headers()
        )

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error"] == "content_routing_unavailable"


# --- #105: the tenant's own `model` setting is the model its run uses ---------------------------
# The real resolver, `resolve_tenant_chat_model`, from the record the request's context
# carries (dev-headers reads it through the installed `FakeControlPlaneReads.records`), a real
# gateway credential file; only client construction (`app.llm.build_tenant_chat_model`, the
# network boundary) is swapped for a `FunctionModel` that records the bare model name it was
# built for and answers with it.

# `eu` allows two chat models here (the shipped config allows one), so a tenant's own choice is
# distinguishable from the deployment default `claude-eu`.
_TWO_EU_MODELS = ResidencyAllowList.from_data(
    {
        "residency": {
            "eu": {
                "model_host_patterns": ["litellm"],
                "embedding_endpoint": "https://gateway-eu.internal/v1",
                "trace_sink_host": "eu.cloud.langfuse.com",
                "models": ["claude-eu", "claude-eu-large", "embeddings"],
            },
            "us": {
                "model_host_patterns": ["gateway-us.internal"],
                "embedding_endpoint": "https://gateway-us.internal/v1",
                "trace_sink_host": "us.cloud.langfuse.com",
                "models": ["claude", "embeddings"],
            },
        }
    }
)


@pytest.fixture
def ran_with(monkeypatch, tmp_path) -> list[str]:
    """Every bare model name a run was actually driven with, in order."""
    from app import llm as llm_module

    (tmp_path / "acme-gateway-key").write_text("sk-acme-secret")
    settings = get_settings().model_copy(
        update={"gateway_credentials_dir": str(tmp_path), "residency_allow_list": _TWO_EU_MODELS}
    )
    monkeypatch.setattr(llm_module, "get_settings", lambda: settings)
    seen: list[str] = []

    def _build(tenant_id, bare_model_name, credential, *, settings=None):
        assert credential.get_secret_value() == "sk-acme-secret"

        async def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(bare_model_name)
            return ModelResponse(parts=[TextPart(content=f"ran with {bare_model_name}")])

        async def _respond_stream(messages: list[ModelMessage], info: AgentInfo):
            seen.append(bare_model_name)
            yield f"ran with {bare_model_name}"

        return FunctionModel(_respond, stream_function=_respond_stream)

    monkeypatch.setattr(llm_module, "build_tenant_chat_model", _build)

    async def _no_history(ctx, conversation_id):
        return []

    monkeypatch.setattr(conversation_tools, "load_conversation_history", _no_history)
    return seen


def _tenant_with(model: str | None, *, residency: str | None = "eu") -> uuid.UUID:
    tenant_id = uuid.uuid4()
    set_default_adapter_for_tests(
        FakeControlPlaneReads(
            records={
                tenant_id: TenantRecord(
                    tenant_id=tenant_id,
                    residency=residency,
                    gateway_credential_alias="acme-gateway-key",
                    settings=TenantSettings(model=model),
                )
            }
        )
    )
    return tenant_id


async def _post_run(client: httpx.AsyncClient, tenant_id: uuid.UUID) -> httpx.Response:
    return await client.post(
        f"/v1/t/{tenant_id}/agents/assistant/run", json={"prompt": "hi"}, headers=_headers()
    )


async def _post_chat(client: httpx.AsyncClient, tenant_id: uuid.UUID) -> httpx.Response:
    return await client.post(
        f"/v1/t/{tenant_id}/api/chat", json=_submit_message_body(), headers=_headers()
    )


async def test_run_uses_the_tenants_own_allow_listed_model_setting(asgi_client, ran_with):
    tenant_id = _tenant_with("claude-eu-large")

    async with asgi_client:
        response = await _post_run(asgi_client, tenant_id)

    assert response.status_code == 200, response.text
    assert response.json()["output"] == "ran with claude-eu-large"
    assert ran_with == ["claude-eu-large"]


async def test_chat_uses_the_tenants_own_allow_listed_model_setting(asgi_client, ran_with):
    tenant_id = _tenant_with("claude-eu-large")

    async with asgi_client:
        response = await _post_chat(asgi_client, tenant_id)

    assert response.status_code == 200, response.text
    assert "ran with claude-eu-large" in response.text
    assert ran_with == ["claude-eu-large"]


async def test_a_tenant_without_a_model_setting_runs_the_deployment_default(asgi_client, ran_with):
    tenant_id = _tenant_with(None)

    async with asgi_client:
        response = await _post_run(asgi_client, tenant_id)

    assert response.status_code == 200, response.text
    assert ran_with == ["claude-eu"]


@pytest.mark.parametrize("post", [_post_run, _post_chat])
async def test_a_model_setting_outside_the_residencys_list_is_a_503_routing_error(
    asgi_client, ran_with, post
):
    """`claude` is the `us` alias: stored for an `eu` tenant (say, the allow-list changed since it
    was written), the request fails closed -- never silently served by the default model."""
    tenant_id = _tenant_with("claude")

    async with asgi_client:
        response = await post(asgi_client, tenant_id)

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error"] == "content_routing_unavailable"
    assert ran_with == []


async def test_a_record_without_a_residency_is_a_503_routing_error(asgi_client, ran_with):
    tenant_id = _tenant_with("claude-eu", residency=None)

    async with asgi_client:
        response = await _post_run(asgi_client, tenant_id)

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error"] == "content_routing_unavailable"
    assert ran_with == []


async def test_run_preparation_fails_closed_for_a_context_without_a_record():
    """A job or test context carries no record: preparing a run refuses it rather than reading
    the control plane again or falling back to the deployment's residency (#105)."""
    ctx = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())
    with pytest.raises(ResidencyUnresolved):
        await prepare_run(ctx)


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

    def _fake_resolve(record, *, settings=None):
        return clients[record.tenant_id]

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

    ctx_eu = RequestContext(
        tenant_id=tenant_eu,
        identity_id=uuid.uuid4(),
        tenant_record=TenantRecord(tenant_id=tenant_eu, residency="eu"),
    )
    ctx_us = RequestContext(
        tenant_id=tenant_us,
        identity_id=uuid.uuid4(),
        tenant_record=TenantRecord(tenant_id=tenant_us, residency="us"),
    )

    await document_tools.search_documents(ctx_eu, "acme contract")
    await document_tools.search_documents(ctx_us, "acme contract")

    assert clients[tenant_eu].embeddings.calls  # the eu client's own embeddings.create() ran
    assert clients[tenant_us].embeddings.calls
    embeddings_seen = [embedding for _session, embedding in captured]
    assert [1.0, 0.0] in embeddings_seen
    assert [0.0, 1.0] in embeddings_seen


@pytest.fixture
def no_tenant_session(monkeypatch):
    @asynccontextmanager
    async def _fake_tenant_session(ctx):
        yield object()

    monkeypatch.setattr(document_tools, "tenant_session", _fake_tenant_session)


async def test_search_documents_fails_closed_when_residency_is_unresolvable(no_tenant_session):
    """#61 AC3: the search path fails exactly like the chat path for a tenant whose record has no
    residency -- never a fallback to a default embedding endpoint (the real resolver, #105)."""
    tenant_id = uuid.uuid4()
    ctx = RequestContext(
        tenant_id=tenant_id,
        identity_id=uuid.uuid4(),
        tenant_record=TenantRecord(tenant_id=tenant_id, gateway_credential_alias="acme"),
    )
    with pytest.raises(ResidencyUnresolved):
        await document_tools.search_documents(ctx, "acme contract")


async def test_search_documents_reads_the_record_for_a_context_without_one(no_tenant_session):
    """The stdio MCP development context carries no record: the tool takes it from the one
    record read (`ControlPlaneReads.get_tenant_record`) and resolves from that -- here the fake
    control plane's residency-less record, so it fails closed."""
    tenant_id = uuid.uuid4()
    asked: list[uuid.UUID] = []

    class _Reads(FakeControlPlaneReads):
        async def get_tenant_record(self, *, tenant_id):
            asked.append(tenant_id)
            return await super().get_tenant_record(tenant_id=tenant_id)

    set_default_adapter_for_tests(_Reads())
    ctx = RequestContext(tenant_id=tenant_id, identity_id=uuid.uuid4())

    with pytest.raises(ResidencyUnresolved):
        await document_tools.search_documents(ctx, "acme contract")
    assert asked == [tenant_id]


# --- AC6: the embeddings module docstring states the one-family / re-embedding-migration rule --


def test_embeddings_docstring_states_one_family_shared_across_residencies():
    import app.embeddings as embeddings_module

    doc = " ".join((embeddings_module.__doc__ or "").lower().split())
    assert "one embedding model family" in doc
    assert "residency" in doc and "serving region" in doc
    assert "re-embedding migration" in doc
    assert "new vector column" in doc
