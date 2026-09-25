"""ASGI-seam tests for tenant suspension (Spec 9 / #69, ADR-0010) under AUTH_MODE=dev-headers --
the mode `tests/test_jwt_auth.py` doesn't cover (that file already exercises AUTH_MODE=jwt's own
inline suspension check, from #24). No real Postgres: `app.tenant_suspension`'s one control-plane
read is faked here, exactly like `tests/test_jwt_auth.py` fakes it for the jwt path, and like this
suite's own `not_suspended` autouse fixture (`tests/conftest.py`) fakes it by default everywhere
else.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

import httpx
import pytest

import app.tenant_suspension as tenant_suspension_module
from app.agents import assistant as assistant_module
from app.main import app


def _suspend(monkeypatch, suspended_tenant_id: uuid.UUID) -> None:
    """Overrides this test's `not_suspended` autouse fixture: `suspended_tenant_id` is reported
    suspended, every other tenant stays unsuspended."""

    @asynccontextmanager
    async def _fake_control_session():
        yield None

    class _FakeTenantAuthSettingsRepository:
        async def get(self, session, *, tenant_id: uuid.UUID, default_issuer=None):
            from types import SimpleNamespace

            if tenant_id != suspended_tenant_id:
                return None
            return SimpleNamespace(issuer=default_issuer, suspended=True)

    monkeypatch.setattr(tenant_suspension_module, "control_session", _fake_control_session)
    monkeypatch.setattr(
        tenant_suspension_module,
        "TenantAuthSettingsRepository",
        _FakeTenantAuthSettingsRepository,
    )


@pytest.fixture
def client(monkeypatch, fake_search, test_model):
    monkeypatch.setattr(assistant_module.document_tools, "search_documents", fake_search)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_dev_headers_rejects_a_suspended_tenant_with_the_generic_forbidden_body(
    client, monkeypatch
):
    import app.deps as deps_module

    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    _suspend(monkeypatch, tenant_id)

    async with client:
        response = await client.post(
            f"/v1/t/{tenant_id}/agents/assistant/run",
            json={"prompt": "Hi"},
            headers={"X-Identity-Id": str(identity_id)},
        )
    assert response.status_code == 403
    assert response.json()["detail"] == deps_module.FORBIDDEN_DETAIL


async def test_dev_headers_chat_endpoint_also_rejects_a_suspended_tenant(client, monkeypatch):
    import app.deps as deps_module

    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    _suspend(monkeypatch, tenant_id)

    async with client:
        response = await client.post(
            f"/v1/t/{tenant_id}/api/chat",
            json={
                "id": "conv-1",
                "trigger": "submit-message",
                "messages": [
                    {"id": "m1", "role": "user", "parts": [{"type": "text", "text": "hi"}]}
                ],
            },
            headers={"X-Identity-Id": str(identity_id)},
        )
    assert response.status_code == 403
    assert response.json()["detail"] == deps_module.FORBIDDEN_DETAIL


async def test_unsuspended_tenant_is_unaffected(client, monkeypatch):
    """A different tenant, never marked suspended, is unaffected by the fake above -- suspension
    is checked per tenant_id, not process-wide."""
    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    other_tenant_id = uuid.uuid4()
    _suspend(monkeypatch, other_tenant_id)

    async with client:
        response = await client.post(
            f"/v1/t/{tenant_id}/agents/assistant/run",
            json={"prompt": "Hi"},
            headers={"X-Identity-Id": str(identity_id)},
        )
    assert response.status_code == 200, response.text
