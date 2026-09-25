"""Per-membership request limit: unit tests for the limiter, and HTTP behavior on the three
agent-facing routes — /agents/assistant/run, /agents/assistant/stream, /api/chat.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from app import config
from app.agents import assistant as assistant_module
from app.main import app
from app.request_limit import RequestLimiter, RequestLimitExceeded, _limiter_for

# --- Unit tests: the limiter itself, no HTTP, no settings ---------------------------------


def test_limiter_allows_up_to_the_configured_max():
    limiter = RequestLimiter(max_requests=3, window_seconds=60.0)
    key = (uuid.uuid4(), uuid.uuid4())
    limiter.check(key, now=0.0)
    limiter.check(key, now=1.0)
    limiter.check(key, now=2.0)
    with pytest.raises(RequestLimitExceeded):
        limiter.check(key, now=3.0)


def test_limiter_resets_after_the_window_elapses():
    limiter = RequestLimiter(max_requests=1, window_seconds=10.0)
    key = (uuid.uuid4(), uuid.uuid4())
    limiter.check(key, now=0.0)
    with pytest.raises(RequestLimitExceeded):
        limiter.check(key, now=5.0)
    # Past the window: the earlier timestamp has aged out.
    limiter.check(key, now=11.0)


def test_limiter_tracks_each_key_independently():
    limiter = RequestLimiter(max_requests=1, window_seconds=60.0)
    key_a, key_b = (uuid.uuid4(), uuid.uuid4()), (uuid.uuid4(), uuid.uuid4())
    limiter.check(key_a, now=0.0)
    with pytest.raises(RequestLimitExceeded):
        limiter.check(key_a, now=0.1)
    # A different (tenant_id, user_id) pair is unaffected by key_a's limit.
    limiter.check(key_b, now=0.1)


def test_exceeded_error_is_distinct_and_labeled():
    exc = RequestLimitExceeded(retry_after_seconds=12.3)
    assert exc.status_code == 429
    assert exc.detail["error"] == "request_limit_exceeded"
    assert exc.detail["retry_after_seconds"] == 12.3


# --- HTTP tests: the ASGI app, real dependency wiring --------------------------------------


@pytest.fixture
def low_limit(monkeypatch):
    """A tiny, deterministic request limit so a handful of requests can exercise it."""
    monkeypatch.setenv("REQUEST_LIMIT_MAX", "2")
    monkeypatch.setenv("REQUEST_LIMIT_WINDOW_SECONDS", "60")
    config.get_settings.cache_clear()
    _limiter_for.cache_clear()
    yield
    config.get_settings.cache_clear()
    _limiter_for.cache_clear()


@pytest.fixture
def client(monkeypatch, fake_search, test_model, low_limit):
    monkeypatch.setattr(assistant_module.document_tools, "search_documents", fake_search)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _headers(tenant_id: uuid.UUID, user_id: uuid.UUID) -> dict[str, str]:
    return {"X-Tenant-Id": str(tenant_id), "X-User-Id": str(user_id)}


async def test_run_endpoint_limits_a_flooding_member(client):
    tenant_id, user_id = uuid.uuid4(), uuid.uuid4()
    headers = _headers(tenant_id, user_id)
    async with client:
        first = await client.post("/agents/assistant/run", json={"prompt": "hi"}, headers=headers)
        second = await client.post("/agents/assistant/run", json={"prompt": "hi"}, headers=headers)
        third = await client.post("/agents/assistant/run", json={"prompt": "hi"}, headers=headers)
    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    body = third.json()["detail"]
    assert body["error"] == "request_limit_exceeded"


async def test_stream_endpoint_limits_a_flooding_member(client):
    tenant_id, user_id = uuid.uuid4(), uuid.uuid4()
    headers = _headers(tenant_id, user_id)
    async with client:
        await client.post("/agents/assistant/stream", json={"prompt": "hi"}, headers=headers)
        await client.post("/agents/assistant/stream", json={"prompt": "hi"}, headers=headers)
        third = await client.post(
            "/agents/assistant/stream", json={"prompt": "hi"}, headers=headers
        )
    assert third.status_code == 429
    assert third.json()["detail"]["error"] == "request_limit_exceeded"


async def test_chat_endpoint_limits_a_flooding_member(client):
    tenant_id, user_id = uuid.uuid4(), uuid.uuid4()
    headers = _headers(tenant_id, user_id)
    payload = {"messages": [], "id": "chat-1"}
    async with client:
        await client.post("/api/chat", json=payload, headers=headers)
        await client.post("/api/chat", json=payload, headers=headers)
        third = await client.post("/api/chat", json=payload, headers=headers)
    assert third.status_code == 429
    assert third.json()["detail"]["error"] == "request_limit_exceeded"


async def test_a_second_membership_is_unaffected(client):
    tenant_id, user_id = uuid.uuid4(), uuid.uuid4()
    other_tenant_id, other_user_id = uuid.uuid4(), uuid.uuid4()
    headers = _headers(tenant_id, user_id)
    other_headers = _headers(other_tenant_id, other_user_id)
    async with client:
        await client.post("/agents/assistant/run", json={"prompt": "hi"}, headers=headers)
        await client.post("/agents/assistant/run", json={"prompt": "hi"}, headers=headers)
        limited = await client.post("/agents/assistant/run", json={"prompt": "hi"}, headers=headers)
        # A different (tenant_id, user_id) pair, same window: unaffected.
        unaffected = await client.post(
            "/agents/assistant/run", json={"prompt": "hi"}, headers=other_headers
        )
    assert limited.status_code == 429
    assert unaffected.status_code == 200


async def test_same_tenant_different_user_is_unaffected(client):
    tenant_id = uuid.uuid4()
    user_id, other_user_id = uuid.uuid4(), uuid.uuid4()
    async with client:
        await client.post(
            "/agents/assistant/run", json={"prompt": "hi"}, headers=_headers(tenant_id, user_id)
        )
        await client.post(
            "/agents/assistant/run", json={"prompt": "hi"}, headers=_headers(tenant_id, user_id)
        )
        limited = await client.post(
            "/agents/assistant/run", json={"prompt": "hi"}, headers=_headers(tenant_id, user_id)
        )
        # Another member of the SAME tenant, same window: unaffected.
        unaffected = await client.post(
            "/agents/assistant/run",
            json={"prompt": "hi"},
            headers=_headers(tenant_id, other_user_id),
        )
    assert limited.status_code == 429
    assert unaffected.status_code == 200
