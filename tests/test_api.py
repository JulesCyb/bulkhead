"""API tests via ASGI, auth in dev-headers mode, search and model replaced."""

from __future__ import annotations

import asyncio
import uuid

import httpx
import pytest
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from app.agents import assistant as assistant_module
from app.config import Settings
from app.main import app
from tests.conftest import (
    looping_tool_calls,
    looping_tool_calls_stream,
    make_stalling_model,
    make_stalling_stream_model,
)

AUTH_HEADERS = {"X-Tenant-Id": str(uuid.uuid4()), "X-User-Id": str(uuid.uuid4())}


@pytest.fixture
def client(monkeypatch, fake_search, test_model):
    # Search without a DB: AssistantDeps resolves the default at runtime -> inject the fake.
    monkeypatch.setattr(assistant_module.document_tools, "search_documents", fake_search)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def raw_client(monkeypatch, fake_search):
    """Like `client`, but without the `test_model` override — for tests that need the model
    production code actually resolves (`get_model()`), so the `run_limits` wiring in
    `app/agents/assistant.py` and `app/api/agents.py` is genuinely exercised rather than bypassed.
    """
    monkeypatch.setattr(assistant_module.document_tools, "search_documents", fake_search)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


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
        response = await client.post("/agents/assistant/run", json={"prompt": "Hi"})
    assert response.status_code == 401


async def test_run_with_context(client, calls):
    tenant_id, user_id = uuid.uuid4(), uuid.uuid4()
    async with client:
        response = await client.post(
            "/agents/assistant/run",
            json={"prompt": "What does the contract say?"},
            headers={"X-Tenant-Id": str(tenant_id), "X-User-Id": str(user_id)},
        )
    assert response.status_code == 200, response.text
    assert response.json()["output"]
    assert calls and calls[0][0] == tenant_id


async def test_run_ends_with_defined_error_when_tool_call_ceiling_exceeded(
    raw_client, small_run_limits, monkeypatch
):
    """One-shot endpoint: an engineered run that calls a tool more times than the configured
    ceiling ends in a defined error status and body, not a raw exception.
    """
    small_run_limits()
    monkeypatch.setattr(
        assistant_module, "get_model", lambda name: FunctionModel(looping_tool_calls)
    )
    async with raw_client:
        response = await raw_client.post(
            "/agents/assistant/run", json={"prompt": "loop please"}, headers=AUTH_HEADERS
        )
    assert response.status_code == 429, response.text
    assert response.json()["detail"]["error"] == "run_limit_exceeded"


async def test_stream_ends_with_terminal_error_event_when_tool_call_ceiling_exceeded(
    raw_client, small_run_limits, monkeypatch
):
    """Streaming endpoint: the same engineered run ends in a terminal `event: error` before the
    stream closes, never a silent stop.
    """
    small_run_limits()
    monkeypatch.setattr(
        assistant_module,
        "get_model",
        lambda name: FunctionModel(looping_tool_calls, stream_function=looping_tool_calls_stream),
    )
    async with raw_client:
        response = await raw_client.post(
            "/agents/assistant/stream", json={"prompt": "loop please"}, headers=AUTH_HEADERS
        )
    assert response.status_code == 200
    assert "event: error" in response.text
    assert "run_limit_exceeded" in response.text
    assert "event: done" not in response.text


async def test_stream_ends_with_distinct_error_when_wall_clock_deadline_exceeded(
    raw_client, small_run_limits, monkeypatch
):
    """A stalled provider (FunctionModel that sleeps) cannot hold the streaming client open past
    the run's own deadline — it ends with a distinct, generic error, and the stream closes,
    instead of waiting for the provider's own (much longer) default timeout.
    """
    small_run_limits(run_deadline_seconds=0.2)
    monkeypatch.setattr(
        assistant_module,
        "get_model",
        lambda name: FunctionModel(
            make_stalling_model(seconds=30), stream_function=make_stalling_stream_model(seconds=30)
        ),
    )
    async with raw_client:
        response = await asyncio.wait_for(
            raw_client.post(
                "/agents/assistant/stream", json={"prompt": "hang please"}, headers=AUTH_HEADERS
            ),
            timeout=5,
        )
    assert response.status_code == 200
    assert "event: error" in response.text
    assert "run_deadline_exceeded" in response.text
    assert "event: done" not in response.text


async def test_run_within_limits_completes_normally(
    raw_client, small_run_limits, calls, monkeypatch
):
    """A run within the ceilings completes normally, unaffected by the new limiting."""
    small_run_limits()
    monkeypatch.setattr(assistant_module, "get_model", lambda name: TestModel())
    async with raw_client:
        response = await raw_client.post(
            "/agents/assistant/run", json={"prompt": "hi"}, headers=AUTH_HEADERS
        )
    assert response.status_code == 200, response.text
    assert response.json()["output"]
