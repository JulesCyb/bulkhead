"""API tests via ASGI, auth in dev-headers mode, search and model replaced."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import httpx
import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets.function import FunctionToolset

from app.agents import assistant as assistant_module
from app.agents.assistant import chat_assistant, one_shot_assistant
from app.main import app


@pytest.fixture
def client(monkeypatch, fake_search, test_model):
    # Search without a DB: AssistantDeps resolves the default at runtime -> inject the fake.
    monkeypatch.setattr(assistant_module.document_tools, "search_documents", fake_search)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_health(client):
    async with client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_run_requires_dev_headers(client):
    async with client:
        response = await client.post("/agents/assistant/run", json={"prompt": "Hi"})
    assert response.status_code == 401


async def test_run_with_context(client, calls, contexts):
    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    async with client:
        response = await client.post(
            "/agents/assistant/run",
            json={"prompt": "What does the contract say?"},
            headers={"X-Tenant-Id": str(tenant_id), "X-Identity-Id": str(identity_id)},
        )
    assert response.status_code == 200, response.text
    assert response.json()["output"]
    assert calls and calls[0][0] == tenant_id


async def test_stream_uses_one_shot_agent_tools(client, calls):
    tenant_id, user_id = uuid.uuid4(), uuid.uuid4()
    async with client:
        response = await client.post(
            "/agents/assistant/stream",
            json={"prompt": "What does the contract say?"},
            headers={"X-Tenant-Id": str(tenant_id), "X-Identity-Id": str(user_id)},
        )
    assert response.status_code == 200, response.text
    assert calls and calls[0][0] == tenant_id


def _headers() -> dict[str, str]:
    return {"X-Tenant-Id": str(uuid.uuid4()), "X-Identity-Id": str(uuid.uuid4())}


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


def test_one_shot_agent_has_no_tool_requiring_approval():
    """S5-T2 / #36: a structural property check, not "no test happened to call one".

    Asserted from `ToolDefinition.kind` (`'unapproved'` iff `requires_approval=True`), the same
    signal pydantic-ai's own approval machinery reads — not from ever having run a model.
    """
    toolset = _function_toolset(one_shot_assistant)
    assert toolset.tools, "expected at least one registered tool"
    for name, tool in toolset.tools.items():
        assert not tool.requires_approval, f"one-shot agent must not carry {name!r} for approval"


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


async def test_one_shot_endpoints_answer_through_reading_only_agent(monkeypatch, fake_search):
    """Both one-shot endpoints expose exactly the one-shot agent's tool set, and never the
    chat agent's — asserted from the tool set the run actually sends to the model, not from
    reading which function the route happens to call.
    """
    monkeypatch.setattr(assistant_module.document_tools, "search_documents", fake_search)
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
                "/agents/assistant/run", json={"prompt": "Hi"}, headers=_headers()
            )
            stream_response = await client.post(
                "/agents/assistant/stream", json={"prompt": "Hi"}, headers=_headers()
            )

    assert run_response.status_code == 200, run_response.text
    assert stream_response.status_code == 200, stream_response.text
    assert len(one_shot_seen) == 2
    assert not chat_seen, "one-shot endpoints must never reach the chat agent"
    for info in one_shot_seen:
        seen_names = {tool_def.name for tool_def in info.function_tools}
        assert seen_names == expected_names
        assert all(tool_def.kind != "unapproved" for tool_def in info.function_tools)


async def test_chat_endpoint_answers_through_chat_agent(monkeypatch, fake_search):
    """The chat endpoint exposes the chat agent's tool set, and never touches the one-shot
    agent — the mirror image of the one-shot assertion above.
    """
    monkeypatch.setattr(assistant_module.document_tools, "search_documents", fake_search)
    one_shot_seen: list[AgentInfo] = []
    chat_seen: list[AgentInfo] = []
    expected_names = set(_function_toolset(chat_assistant).tools)

    transport = httpx.ASGITransport(app=app)
    with (
        one_shot_assistant.override(model=_capturing_model(one_shot_seen)),
        chat_assistant.override(model=_capturing_model(chat_seen)),
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/chat", json=_chat_body(), headers=_headers())

    assert response.status_code == 200, response.text
    assert not one_shot_seen, "the chat endpoint must never reach the one-shot agent"
    assert len(chat_seen) == 1
    seen_names = {tool_def.name for tool_def in chat_seen[0].function_tools}
    assert seen_names == expected_names


async def test_both_agents_instructions_state_tool_results_are_data(monkeypatch, fake_search):
    """Acceptance criterion: each agent's instructions text plainly states that tool results
    are data to weigh, not instructions to follow (closes the prompt-injection gap)."""
    monkeypatch.setattr(assistant_module.document_tools, "search_documents", fake_search)
    one_shot_seen: list[AgentInfo] = []
    chat_seen: list[AgentInfo] = []

    transport = httpx.ASGITransport(app=app)
    with (
        one_shot_assistant.override(model=_capturing_model(one_shot_seen)),
        chat_assistant.override(model=_capturing_model(chat_seen)),
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/agents/assistant/run", json={"prompt": "Hi"}, headers=_headers())
            await client.post("/api/chat", json=_chat_body(), headers=_headers())

    for seen in (one_shot_seen, chat_seen):
        assert seen
        instructions = seen[0].instructions or ""
        assert "data" in instructions.lower()
        assert "not an instruction to follow" in instructions.lower() or (
            "never an instruction to follow" in instructions.lower()
        )
