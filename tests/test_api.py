"""API tests via ASGI, auth in dev-headers mode, search and model replaced."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

import httpx
import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets.function import FunctionToolset

from app.agents.assistant import chat_assistant, one_shot_assistant
from app.config import Settings
from app.main import app
from app.tools import conversations as conversation_tools
from tests.conftest import (
    looping_tool_calls,
    looping_tool_calls_stream,
    make_stalling_model,
    make_stalling_stream_model,
)


@pytest.fixture
def client(route_run):
    # Model, search, and conversation history without a DB or a gateway: installed for every run
    # the routes prepare (`app.agents.run`), nothing patched.
    route_run(TestModel(call_tools=["search_documents"]))
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def raw_client(route_run):
    """Like `client`, but with no model installed -- each test installs its own through
    `route_run`, so the run-limit wiring in `app/agents/run.py` is genuinely exercised."""
    route_run()
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def chat_history(monkeypatch, fake_history):
    """The chat route still builds its own tool dependencies until #108 moves it onto
    `app.agents.run`: its history loader is the real module's, replaced here."""
    monkeypatch.setattr(conversation_tools, "load_conversation_history", fake_history)


@pytest.fixture
def small_run_limits(monkeypatch):
    """Configure a tiny run-limit ceiling and deadline, read by `build_run_limits()` at every
    entry point (it calls `app.run_limits.get_settings()` with no arguments).
    """

    def _apply(**overrides):
        settings = Settings(run_tool_calls_limit=2, run_request_limit=50, **overrides)
        monkeypatch.setattr("app.run_limits.get_settings", lambda: settings)
        return settings

    return _apply


async def test_health(client):
    async with client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_run_requires_dev_headers(client):
    async with client:
        response = await client.post(
            f"/v1/t/{uuid.uuid4()}/agents/assistant/run", json={"prompt": "Hi"}
        )
    assert response.status_code == 401


async def test_run_with_context(client, calls, contexts):
    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    async with client:
        response = await client.post(
            f"/v1/t/{tenant_id}/agents/assistant/run",
            json={"prompt": "What does the contract say?"},
            headers={"X-Identity-Id": str(identity_id)},
        )
    assert response.status_code == 200, response.text
    assert response.json()["output"]
    assert calls and calls[0][0] == tenant_id


async def test_run_ends_with_defined_error_when_tool_call_ceiling_exceeded(
    raw_client, small_run_limits, route_run
):
    """One-shot endpoint: an engineered run that calls a tool more times than the configured
    ceiling ends in a defined error status and body, not a raw exception.
    """
    small_run_limits()
    route_run(FunctionModel(looping_tool_calls))
    async with raw_client:
        response = await raw_client.post(
            _tenant_path("/agents/assistant/run"),
            json={"prompt": "loop please"},
            headers=_headers(),
        )
    assert response.status_code == 429, response.text
    assert response.json()["detail"]["error"] == "run_limit_exceeded"


async def test_stream_ends_with_terminal_error_event_when_tool_call_ceiling_exceeded(
    raw_client, small_run_limits, route_run
):
    """Streaming endpoint: the same engineered run ends in a terminal `event: error` before the
    stream closes, never a silent stop.
    """
    small_run_limits()
    route_run(FunctionModel(looping_tool_calls, stream_function=looping_tool_calls_stream))
    async with raw_client:
        response = await raw_client.post(
            _tenant_path("/agents/assistant/stream"),
            json={"prompt": "loop please"},
            headers=_headers(),
        )
    assert response.status_code == 200
    assert "event: error" in response.text
    assert "run_limit_exceeded" in response.text
    assert "event: done" not in response.text


async def test_stream_ends_with_distinct_error_when_wall_clock_deadline_exceeded(
    raw_client, small_run_limits, route_run
):
    """A stalled provider (FunctionModel that sleeps) cannot hold the streaming client open past
    the run's own deadline — it ends with a distinct, generic error, and the stream closes,
    instead of waiting for the provider's own (much longer) default timeout.
    """
    small_run_limits(run_deadline_seconds=0.2)
    route_run(
        FunctionModel(
            make_stalling_model(seconds=30),
            stream_function=make_stalling_stream_model(seconds=30),
        )
    )
    async with raw_client:
        response = await asyncio.wait_for(
            raw_client.post(
                _tenant_path("/agents/assistant/stream"),
                json={"prompt": "hang please"},
                headers=_headers(),
            ),
            timeout=5,
        )
    assert response.status_code == 200
    assert "event: error" in response.text
    assert "run_deadline_exceeded" in response.text
    assert "event: done" not in response.text


async def test_run_within_limits_completes_normally(raw_client, small_run_limits, calls, route_run):
    """A run within the ceilings completes normally, unaffected by the new limiting."""
    small_run_limits()
    route_run(TestModel())
    async with raw_client:
        response = await raw_client.post(
            _tenant_path("/agents/assistant/run"), json={"prompt": "hi"}, headers=_headers()
        )
    assert response.status_code == 200, response.text
    assert response.json()["output"]


async def test_run_ignores_a_stray_tenant_header_and_uses_the_path(client, calls):
    """ADR-0012: the URL path is the request's sole statement of intent. A stray old-style
    X-Tenant-Id header naming a different tenant must be ignored entirely."""
    path_tenant_id, header_tenant_id, identity_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    async with client:
        response = await client.post(
            f"/v1/t/{path_tenant_id}/agents/assistant/run",
            json={"prompt": "What does the contract say?"},
            headers={"X-Identity-Id": str(identity_id), "X-Tenant-Id": str(header_tenant_id)},
        )
    assert response.status_code == 200, response.text
    assert calls and calls[0][0] == path_tenant_id
    assert calls[0][0] != header_tenant_id


async def test_chat_old_unprefixed_path_is_gone(client):
    async with client:
        response = await client.post("/api/chat", json=_chat_body(), headers=_headers())
    assert response.status_code == 404


async def test_stream_uses_one_shot_agent_tools(client, calls):
    tenant_id, user_id = uuid.uuid4(), uuid.uuid4()
    async with client:
        response = await client.post(
            f"/v1/t/{tenant_id}/agents/assistant/stream",
            json={"prompt": "What does the contract say?"},
            headers={"X-Identity-Id": str(user_id)},
        )
    assert response.status_code == 200, response.text
    assert calls and calls[0][0] == tenant_id


def _headers() -> dict[str, str]:
    return {"X-Identity-Id": str(uuid.uuid4())}


def _tenant_path(suffix: str) -> str:
    return f"/v1/t/{uuid.uuid4()}{suffix}"


def _chat_body(text: str = "What does the contract say?") -> dict:
    return {
        "id": "conv-1",
        "trigger": "submit-message",
        "messages": [
            {"id": "m1", "role": "user", "parts": [{"type": "text", "text": text}]},
        ],
    }


def _function_toolset(agent) -> FunctionToolset:
    for toolset in agent.toolsets:
        if isinstance(toolset, FunctionToolset):
            return toolset
    raise AssertionError(f"{agent} has no function toolset")


# The structural ADR-0007 guarantee (the reading-only agent registers no tool with an approval
# validator, and `answer`/`stream_text` bind it) is asserted once, at the run module's own seam:
# `tests/test_run.py::test_answer_and_stream_text_bind_the_reading_only_agent`. The route-level
# test below shows the one-shot endpoints inherit it.


def test_one_shot_and_chat_are_distinct_agents():
    assert one_shot_assistant is not chat_assistant


def _capturing_model(sink: list[AgentInfo]) -> FunctionModel:
    async def call(messages: list, info: AgentInfo) -> ModelResponse:
        sink.append(info)
        return ModelResponse(parts=[TextPart("ok")])

    async def stream_call(messages: list, info: AgentInfo) -> AsyncIterator[str]:
        sink.append(info)
        yield "ok"

    return FunctionModel(call, stream_function=stream_call)


async def test_one_shot_endpoints_answer_through_reading_only_agent(route_run):
    """Both one-shot endpoints expose exactly the one-shot agent's tool set, and never the
    chat agent's — asserted from the tool set the run actually sends to the model, not from
    reading which function the route happens to call.
    """
    route_run(TestModel())
    one_shot_seen: list[AgentInfo] = []
    chat_seen: list[AgentInfo] = []
    expected_names = set(_function_toolset(one_shot_assistant).tools)

    transport = httpx.ASGITransport(app=app)
    with (
        one_shot_assistant.override(model=_capturing_model(one_shot_seen)),
        chat_assistant.override(model=_capturing_model(chat_seen)),
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            run_response = await client.post(
                _tenant_path("/agents/assistant/run"), json={"prompt": "Hi"}, headers=_headers()
            )
            stream_response = await client.post(
                _tenant_path("/agents/assistant/stream"), json={"prompt": "Hi"}, headers=_headers()
            )

    assert run_response.status_code == 200, run_response.text
    assert stream_response.status_code == 200, stream_response.text
    assert len(one_shot_seen) == 2
    assert not chat_seen, "one-shot endpoints must never reach the chat agent"
    for info in one_shot_seen:
        seen_names = {tool_def.name for tool_def in info.function_tools}
        assert seen_names == expected_names
        assert all(tool_def.kind != "unapproved" for tool_def in info.function_tools)


async def test_chat_endpoint_answers_through_chat_agent(route_run, chat_history):
    """The chat endpoint exposes the chat agent's tool set, and never touches the one-shot
    agent — the mirror image of the one-shot assertion above.
    """
    route_run(TestModel())
    one_shot_seen: list[AgentInfo] = []
    chat_seen: list[AgentInfo] = []
    expected_names = set(_function_toolset(chat_assistant).tools)

    transport = httpx.ASGITransport(app=app)
    with (
        one_shot_assistant.override(model=_capturing_model(one_shot_seen)),
        chat_assistant.override(model=_capturing_model(chat_seen)),
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                _tenant_path("/api/chat"), json=_chat_body(), headers=_headers()
            )

    assert response.status_code == 200, response.text
    assert not one_shot_seen, "the chat endpoint must never reach the one-shot agent"
    assert len(chat_seen) == 1
    seen_names = {tool_def.name for tool_def in chat_seen[0].function_tools}
    assert seen_names == expected_names


async def test_both_agents_instructions_state_tool_results_are_data(route_run, chat_history):
    """Acceptance criterion: each agent's instructions text plainly states that tool results
    are data to weigh, not instructions to follow (closes the prompt-injection gap)."""
    route_run(TestModel())
    one_shot_seen: list[AgentInfo] = []
    chat_seen: list[AgentInfo] = []

    transport = httpx.ASGITransport(app=app)
    with (
        one_shot_assistant.override(model=_capturing_model(one_shot_seen)),
        chat_assistant.override(model=_capturing_model(chat_seen)),
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                _tenant_path("/agents/assistant/run"), json={"prompt": "Hi"}, headers=_headers()
            )
            await client.post(_tenant_path("/api/chat"), json=_chat_body(), headers=_headers())

    for seen in (one_shot_seen, chat_seen):
        assert seen
        instructions = seen[0].instructions or ""
        assert "data" in instructions.lower()
        assert "not an instruction to follow" in instructions.lower() or (
            "never an instruction to follow" in instructions.lower()
        )


def test_no_response_schema_exposes_isolation_tier_or_database_alias():
    """Acceptance criterion (#75, ADR-0002): a tenant's isolation tier and database alias are
    control-plane facts read only inside `tenant_session()`'s own routing -- no route's request
    or response schema anywhere in the API may leak either to a client. Walks the whole OpenAPI
    schema (every route's request/response models, not just the ones this file happens to
    exercise) rather than checking one endpoint by name.
    """
    schema = app.openapi()
    forbidden = {"isolation_tier", "database_alias"}
    for name, definition in schema.get("components", {}).get("schemas", {}).items():
        properties = definition.get("properties", {})
        leaked = forbidden & properties.keys()
        assert not leaked, f"schema {name!r} exposes forbidden field(s) {leaked}"
