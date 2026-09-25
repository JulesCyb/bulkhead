"""ASGI tests: every authenticated response carries the context's request id (S4-T1, #31).

Covers: the run, stream, and chat endpoints all carry X-Request-Id matching the context that
served the request; the one-shot endpoints' shape (no conversation id, existing response shape)
is guarded as a regression ahead of the rest of Spec 4's changes to the chat endpoint; and a
request missing identity/tenant context still fails with its existing 401 before header logic
runs.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
from pydantic_ai.models.test import TestModel

from app.agents import assistant as assistant_module
from app.api import chat as chat_module
from app.context import RequestContext
from app.main import app

HEADER = "X-Request-Id"


@pytest.fixture
def captured_ctx() -> list[RequestContext]:
    return []


@pytest.fixture
def client(monkeypatch, captured_ctx, test_model):
    async def _search(ctx: RequestContext, query: str, limit: int):
        captured_ctx.append(ctx)
        return []

    monkeypatch.setattr(assistant_module.document_tools, "search_documents", _search)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _headers() -> dict[str, str]:
    return {"X-Identity-Id": str(uuid.uuid4())}


def _tenant_path(suffix: str) -> str:
    return f"/v1/t/{uuid.uuid4()}{suffix}"


async def test_run_endpoint_carries_request_id_header(client, captured_ctx):
    async with client:
        response = await client.post(
            _tenant_path("/agents/assistant/run"), json={"prompt": "Hi"}, headers=_headers()
        )
    assert response.status_code == 200, response.text
    assert captured_ctx
    assert response.headers[HEADER] == captured_ctx[0].request_id


async def test_stream_endpoint_carries_request_id_header(client, captured_ctx):
    async with client:
        async with client.stream(
            "POST",
            _tenant_path("/agents/assistant/stream"),
            json={"prompt": "Hi"},
            headers=_headers(),
        ) as response:
            assert response.status_code == 200
            header_value = response.headers[HEADER]
            async for _ in response.aiter_bytes():
                pass
    assert captured_ctx
    assert header_value == captured_ctx[0].request_id


async def test_chat_endpoint_carries_request_id_header(monkeypatch, fake_history):
    from tests.conftest import resolve_to_model

    monkeypatch.setattr(chat_module, "resolve_chat_model", resolve_to_model(TestModel()))
    monkeypatch.setattr(
        assistant_module.conversation_tools, "load_conversation_history", fake_history
    )
    transport = httpx.ASGITransport(app=app)
    body = {
        "id": "conv-1",
        "trigger": "submit-message",
        "messages": [{"id": "m1", "role": "user", "parts": [{"type": "text", "text": "Hi"}]}],
    }
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(_tenant_path("/api/chat"), json=body, headers=_headers())
    assert response.status_code == 200, response.text
    assert response.headers.get(HEADER)


async def test_one_shot_endpoints_stay_stateless_shape(client, captured_ctx):
    """Regression guard for ADR-0007: the one-shot endpoints still take no conversation id
    and their response shape (`{"output": ...}` only) is unchanged by this header addition."""
    async with client:
        run_response = await client.post(
            _tenant_path("/agents/assistant/run"),
            json={"prompt": "What does the contract say?"},
            headers=_headers(),
        )
        stream_headers = _headers()
        async with client.stream(
            "POST",
            _tenant_path("/agents/assistant/stream"),
            json={"prompt": "What does the contract say?"},
            headers=stream_headers,
        ) as stream_response:
            assert stream_response.status_code == 200
            async for _ in stream_response.aiter_bytes():
                pass

    assert run_response.status_code == 200, run_response.text
    body = run_response.json()
    assert set(body.keys()) == {"output"}
    assert body["output"]
    # Neither endpoint requires or accepts a conversation id.
    assert "conversation_id" not in body


async def test_missing_context_fails_before_header_logic(client):
    async with client:
        response = await client.post(_tenant_path("/agents/assistant/run"), json={"prompt": "Hi"})
    assert response.status_code == 401
    assert HEADER not in response.headers
