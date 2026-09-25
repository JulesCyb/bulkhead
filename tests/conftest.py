"""Shared fixtures: context, fake search, TestModel — no real model call, no DB."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCalls
from pydantic_ai.models.function import DeltaToolCall as _DeltaToolCall
from pydantic_ai.models.test import TestModel

from app.agents.assistant import AssistantDeps, assistant
from app.context import RequestContext
from app.repositories.documents import DocumentHit


@pytest.fixture
def ctx() -> RequestContext:
    return RequestContext(tenant_id=uuid.uuid4(), user_id=uuid.uuid4(), roles=frozenset({"member"}))


@pytest.fixture
def calls() -> list[tuple[uuid.UUID, str, int]]:
    return []


@pytest.fixture
def fake_search(calls):
    async def _search(ctx: RequestContext, query: str, limit: int) -> list[DocumentHit]:
        calls.append((ctx.tenant_id, query, limit))
        return [DocumentHit(id=uuid.uuid4(), title="Acme contract", snippet="…", score=0.9)]

    return _search


@pytest.fixture
def deps(ctx, fake_search) -> AssistantDeps:
    return AssistantDeps(ctx=ctx, search=fake_search)


@pytest.fixture
def test_model():
    """TestModel calls every tool once and answers deterministically."""
    with assistant.override(model=TestModel()):
        yield


def looping_tool_calls(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """An engineered run: calls `search_documents` again on every step, forever — the kind of
    loop a poisoned document can provoke. Used to drive an agent run past its tool-call ceiling.
    """
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name="search_documents",
                args={"query": "again"},
                tool_call_id=f"call-{len(messages)}",
            )
        ]
    )


async def looping_tool_calls_stream(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[DeltaToolCalls]:
    """Streamed counterpart of `looping_tool_calls`, for `FunctionModel(stream_function=...)`."""
    yield {0: _DeltaToolCall(name="search_documents", json_args='{"query": "again"}')}


def make_stalling_model(seconds: float = 10.0):
    """A `FunctionModel` function that never returns within any run's configured deadline —
    simulates a stalled provider without a real network call or a real sleep beyond `seconds`
    (the caller picks a short one; the run's own deadline is expected to fire first).
    """

    async def _stall(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        await asyncio.sleep(seconds)
        return ModelResponse(parts=[TextPart(content="unreachable")])  # pragma: no cover

    return _stall


def make_stalling_stream_model(seconds: float = 10.0):
    """Streamed counterpart of `make_stalling_model`, for `FunctionModel(stream_function=...)`."""

    async def _stall(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        await asyncio.sleep(seconds)
        yield "unreachable"  # pragma: no cover

    return _stall
