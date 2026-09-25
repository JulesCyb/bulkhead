"""API tests via ASGI, auth in dev-headers mode, search and model replaced."""

from __future__ import annotations

import uuid

import httpx
import pytest

from app.agents import assistant as assistant_module
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
    assert contexts and contexts[0].identity_id == identity_id
