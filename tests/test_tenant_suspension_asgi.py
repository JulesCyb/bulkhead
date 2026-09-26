"""ASGI-seam tests for tenant suspension (#106, ADR-0010) under AUTH_MODE=dev-headers -- the mode
`tests/test_jwt_auth.py` doesn't cover (that file already exercises AUTH_MODE=jwt's own path
through `app.context_resolution.resolve_bearer_context`).

Proves the first of suspension's two enforcement points project-wide (`app/db/session.py`'s module
docstring): context resolution (`app.context_resolution.resolve_dev_headers_context`) reads the
tenant record, refuses a suspended one with the generic 403 body, and does this *before* the route
itself ever runs -- proven here by the injected tool (`fake_search`, via the `calls` fixture) never
being invoked at all, standing in for "zero tenant-table statements" in this no-real-Postgres
corner of the suite (the literal statement count, against a real engine, is
`tests/test_tenant_record_integration.py`'s and `tests/test_tenant_session_routing_integration.py`'s
job). The second enforcement point -- `tenant_session()`'s own routing read, for a context that
carries no record at all -- is proven by `tests/test_assistant_suspension.py` (the agent-run entry
points) and `tests/test_mcp_context.py` (the MCP `stdio` fallback), and, against a real database,
by `tests/test_tenant_session_routing_integration.py`.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
from pydantic_ai.models.test import TestModel

from app.main import app
from app.token_verifier import set_default_adapter_for_tests
from tests.conftest import FakeControlPlaneReads


def _suspend(suspended_tenant_id: uuid.UUID) -> None:
    """Overrides this test's `default_control_plane_reads` autouse fixture: `suspended_tenant_id`
    is reported suspended, every other tenant stays unsuspended."""
    set_default_adapter_for_tests(
        FakeControlPlaneReads(auth_settings={suspended_tenant_id: (None, True)})
    )


@pytest.fixture
def client(route_run):
    route_run(TestModel(call_tools=["search_documents"]))
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_dev_headers_rejects_a_suspended_tenant_with_the_generic_forbidden_body(
    client, calls
):
    import app.deps as deps_module

    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    _suspend(tenant_id)

    async with client:
        response = await client.post(
            f"/v1/t/{tenant_id}/agents/assistant/run",
            json={"prompt": "Hi"},
            headers={"X-Identity-Id": str(identity_id)},
        )
    assert response.status_code == 403
    assert response.json()["detail"] == deps_module.FORBIDDEN_DETAIL
    # Context resolution refused the request before the route -- and therefore the tool -- ever
    # ran, standing in for "zero tenant-table statements" (module docstring).
    assert calls == []


async def test_dev_headers_chat_endpoint_also_rejects_a_suspended_tenant(client, calls):
    import app.deps as deps_module

    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    _suspend(tenant_id)

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
    assert calls == []


async def test_unsuspended_tenant_is_unaffected(client, calls):
    """A different tenant, never marked suspended, is unaffected by the fake above -- suspension
    is checked per tenant_id, not process-wide -- and the request proceeds all the way to the
    tool."""
    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    other_tenant_id = uuid.uuid4()
    _suspend(other_tenant_id)

    async with client:
        response = await client.post(
            f"/v1/t/{tenant_id}/agents/assistant/run",
            json={"prompt": "Hi"},
            headers={"X-Identity-Id": str(identity_id)},
        )
    assert response.status_code == 200, response.text
    assert calls and calls[0][0] == tenant_id
