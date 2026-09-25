"""The Vercel AI SDK chat adapter (/v1/t/{tenant_id}/api/chat) picks up the same run limits as
the other two agent entry points, via ASGI — search and model replaced, no real model call.
"""

from __future__ import annotations

import asyncio
import uuid

import httpx
import pytest
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from app.agents import assistant as assistant_module
from app.api import chat as chat_module
from app.config import Settings
from app.main import app
from tests.conftest import (
    looping_tool_calls,
    looping_tool_calls_stream,
    make_stalling_model,
    make_stalling_stream_model,
    resolve_to_model,
)


def _headers() -> dict[str, str]:
    return {"X-Identity-Id": str(uuid.uuid4())}


def _chat_path() -> str:
    return f"/v1/t/{uuid.uuid4()}/api/chat"


def _submit_message_body(text: str = "hi") -> dict:
    return {
        "id": "conv-1",
        "trigger": "submit-message",
        "messages": [{"id": "m1", "role": "user", "parts": [{"type": "text", "text": text}]}],
    }


@pytest.fixture
def client(monkeypatch, fake_search):
    monkeypatch.setattr(assistant_module.document_tools, "search_documents", fake_search)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def small_run_limits(monkeypatch):
    def _apply(**overrides):
        settings = Settings(run_tool_calls_limit=2, run_request_limit=50, **overrides)
        monkeypatch.setattr("app.run_limits.get_settings", lambda: settings)
        return settings

    return _apply


async def test_chat_maps_tool_call_ceiling_to_an_error_chunk(client, small_run_limits, monkeypatch):
    """The same engineered run driven through the chat adapter produces a mapped error rather
    than an unhandled exception."""
    small_run_limits()
    monkeypatch.setattr(
        chat_module,
        "resolve_chat_model",
        resolve_to_model(
            FunctionModel(looping_tool_calls, stream_function=looping_tool_calls_stream)
        ),
    )
    async with client:
        response = await client.post(_chat_path(), json=_submit_message_body(), headers=_headers())
    assert response.status_code == 200, response.text
    assert '"type":"error"' in response.text
    assert response.text.strip().endswith("data: [DONE]")


async def test_chat_maps_wall_clock_deadline_to_a_distinct_error(
    client, small_run_limits, monkeypatch
):
    """A stalled provider cannot hold the chat stream open past the run's own deadline."""
    small_run_limits(run_deadline_seconds=0.2)
    monkeypatch.setattr(
        chat_module,
        "resolve_chat_model",
        resolve_to_model(
            FunctionModel(
                make_stalling_model(seconds=30),
                stream_function=make_stalling_stream_model(seconds=30),
            )
        ),
    )
    async with client:
        response = await asyncio.wait_for(
            client.post(_chat_path(), json=_submit_message_body(), headers=_headers()),
            timeout=5,
        )
    assert response.status_code == 200, response.text
    assert '"type":"error"' in response.text
    assert "wall-clock deadline" in response.text


async def test_chat_within_limits_completes_normally(client, small_run_limits, monkeypatch, calls):
    """A run within the ceilings completes normally, unaffected by the new limiting."""
    small_run_limits()
    monkeypatch.setattr(chat_module, "resolve_chat_model", resolve_to_model(TestModel()))
    async with client:
        response = await client.post(_chat_path(), json=_submit_message_body(), headers=_headers())
    assert response.status_code == 200, response.text
    assert '"type":"error"' not in response.text
