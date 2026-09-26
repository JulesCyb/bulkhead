"""The tenant record without a database (#104): what the HTTP adapter answers and what a tool
receives when the one shared `FakeControlPlaneReads` hands out a record, and how the session layer
treats a record it is given. The database-backed counterpart -- one view read per request,
counted -- is `tests/test_tenant_record_integration.py`.
"""

from __future__ import annotations

import dataclasses
import time
import uuid
from datetime import UTC, datetime

import httpx
import jwt
import pytest

import app.deps as deps_module
from app.agents import assistant as assistant_module
from app.config import Settings
from app.context import RequestContext
from app.context_resolution import (
    ContextRejection,
    RejectionReason,
    RejectionStatus,
    resolve_bearer_context,
)
from app.db.session import TenantSuspendedError, tenant_session
from app.main import app
from app.tenant_record import TenantRecord
from app.tenant_settings import TenantSettings
from app.token_verifier import set_default_adapter_for_tests
from tests.conftest import FakeControlPlaneReads

_EVERY_OTHER_READ = frozenset(
    {"find_identity_by_issuer_and_subject", "get_tenant_auth_settings", "get_membership_role"}
)


async def _post_run(tenant_id: uuid.UUID, identity_id: uuid.UUID) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            f"/v1/t/{tenant_id}/agents/assistant/run",
            json={"prompt": "Hi"},
            headers={"X-Identity-Id": str(identity_id), "X-Roles": "member"},
        )


async def test_a_suspended_record_is_answered_with_the_generic_403_and_nothing_else_is_read():
    tenant_id = uuid.uuid4()
    set_default_adapter_for_tests(
        FakeControlPlaneReads(
            records={
                tenant_id: TenantRecord(
                    tenant_id=tenant_id, suspended_at=datetime(2026, 9, 1, tzinfo=UTC)
                )
            },
            explode=_EVERY_OTHER_READ,
        )
    )

    response = await _post_run(tenant_id, uuid.uuid4())

    assert response.status_code == 403
    assert response.json() == {"detail": deps_module.FORBIDDEN_DETAIL}


async def test_the_tool_receives_the_record_the_context_was_resolved_with(
    monkeypatch, fake_search, test_model, contexts
):
    tenant_id = uuid.uuid4()
    record = TenantRecord(
        tenant_id=tenant_id,
        residency="eu",
        gateway_credential_alias="gw-acme",
        settings=TenantSettings(retention_days=30),
    )
    set_default_adapter_for_tests(FakeControlPlaneReads(records={tenant_id: record}))
    monkeypatch.setattr(assistant_module.document_tools, "search_documents", fake_search)

    response = await _post_run(tenant_id, uuid.uuid4())

    assert response.status_code == 200, response.text
    assert contexts[0].tenant_record == record


async def test_a_bearer_request_for_a_suspended_record_is_refused_before_the_identity_lookup():
    """The bearer chain reads the record right after the token verifies and refuses a suspended
    one there: neither the identity nor the membership is ever looked up."""
    tenant_id = uuid.uuid4()
    secret = "tenant-record-unit-secret-at-least-32-bytes-long"
    settings = Settings(
        _env_file=None,
        environment="test",
        auth_mode="jwt",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        default_identity_issuer="https://idp.example.com",
        jwt_verification_key=secret,
        jwt_algorithm="HS256",
    )
    now = int(time.time())
    token = jwt.encode(
        {
            "iss": "https://idp.example.com",
            "sub": "sub-1",
            "aud": str(tenant_id),
            "iat": now,
            "exp": now + 300,
        },
        secret,
        algorithm="HS256",
    )
    adapter = FakeControlPlaneReads(
        records={tenant_id: TenantRecord(tenant_id=tenant_id, suspended_at=datetime.now(UTC))},
        explode=frozenset({"find_identity_by_issuer_and_subject", "get_membership_role"}),
    )

    outcome = await resolve_bearer_context(
        tenant_id=tenant_id,
        authorization=f"Bearer {token}",
        settings=settings,
        adapter=adapter,
    )

    assert isinstance(outcome, ContextRejection)
    assert (outcome.status, outcome.reason, outcome.detail) == (
        RejectionStatus.FORBIDDEN,
        RejectionReason.TENANT_SUSPENDED,
        deps_module.FORBIDDEN_DETAIL,
    )
    assert outcome.issuer == "https://idp.example.com"


def test_the_record_is_immutable():
    record = TenantRecord(tenant_id=uuid.uuid4(), settings=TenantSettings(model="eu-default"))

    with pytest.raises(dataclasses.FrozenInstanceError):
        record.residency = "us"  # type: ignore[misc]
    with pytest.raises(ValueError):
        record.settings.model = "anything-else"  # type: ignore[misc]


async def test_the_session_layer_refuses_a_suspended_record_without_reading_anything():
    tenant_id = uuid.uuid4()
    ctx = RequestContext(
        tenant_id=tenant_id,
        identity_id=uuid.uuid4(),
        tenant_record=TenantRecord(tenant_id=tenant_id, suspended_at=datetime.now(UTC)),
    )

    with pytest.raises(TenantSuspendedError):
        async with tenant_session(ctx):
            pass


async def test_the_session_layer_refuses_a_record_of_another_tenant():
    ctx = RequestContext(
        tenant_id=uuid.uuid4(),
        identity_id=uuid.uuid4(),
        tenant_record=TenantRecord(tenant_id=uuid.uuid4()),
    )

    with pytest.raises(RuntimeError, match="carries the record of tenant"):
        async with tenant_session(ctx):
            pass
