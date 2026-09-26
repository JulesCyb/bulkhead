"""ASGI-seam tests for an agent identity calling a writing tool under a standing grant (ADR-0007,
Spec 5 / #42), against PostgreSQL + pgvector (`pgserver`, via `uv sync --group dbtest`). Pattern:
`tests/test_writing_tool_approval_integration.py`, driven end to end through
`POST /v1/t/{tenant_id}/api/chat` with a `FunctionModel` standing in for the model -- no real
model call.

Unlike a member's write, an agent identity's call to `require_approval()` (`app/tools/
approvals.py`) never defers: the `membership.role == "agent"` branch either denies outright (no
standing grant covers this tool) or authorizes execution immediately (an active grant does) --
there is no pending action and no second, resumed request either way. So every test here is a
single `POST /api/chat`, not a propose/resume pair.
"""

from __future__ import annotations

import json
import uuid

import httpx
import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.context import RequestContext
from app.db.session import tenant_session
from app.main import app
from app.repositories.standing_grants import StandingGrantRepository

pgserver = pytest.importorskip("pgserver")

from tests.support import (  # noqa: E402
    cluster,
    environment,
    seed_conversation,
    seed_document,
    seed_tenant,
)

_ = (cluster, environment)

CONVERSATION_ID = "conv-agent-1"
TOOL_CALL_ID = "call-rename-agent-1"
ORIGINAL_TITLE = "Original Title"
NEW_TITLE = "Renamed Title"
TOOL_NAME = "rename_document"


async def _document_title(url: str, *, document_id: uuid.UUID) -> str:
    engine = create_async_engine(url)
    async with engine.connect() as conn:
        title = (
            await conn.execute(
                text("SELECT title FROM documents WHERE id = :id"), {"id": document_id}
            )
        ).scalar_one()
    await engine.dispose()
    return title


async def _create_standing_grant_via_repository(
    environment,
    *,
    tenant_id: uuid.UUID,
    agent_membership_id: uuid.UUID,
    granted_by: uuid.UUID,
    tool_name: str = TOOL_NAME,
) -> uuid.UUID:
    """Through the real repository (`app/repositories/standing_grants.py`), exactly like
    `tests/test_standing_grants_integration.py` -- no raw INSERT of its own."""
    ctx = RequestContext(tenant_id=tenant_id, identity_id=granted_by, roles=frozenset({"admin"}))
    async with tenant_session(ctx) as session:
        grant = await StandingGrantRepository().create(
            session,
            ctx,
            agent_membership_id=agent_membership_id,
            tool_name=tool_name,
            granted_by=granted_by,
        )
        return grant.id


async def _pending_action_rows(url: str, *, tenant_id: uuid.UUID) -> list[dict]:
    engine = create_async_engine(url)
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    text("SELECT id FROM pending_actions WHERE tenant_id = :tid"),
                    {"tid": tenant_id},
                )
            )
            .mappings()
            .all()
        )
    await engine.dispose()
    return [dict(row) for row in rows]


async def _audit_events_for_tenant(url: str, *, tenant_id: uuid.UUID) -> list[dict]:
    engine = create_async_engine(url)
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    text(
                        "SELECT kind, standing_grant_id, pending_action_id "
                        "FROM approval_audit_events WHERE tenant_id = :tid ORDER BY seq"
                    ),
                    {"tid": tenant_id},
                )
            )
            .mappings()
            .all()
        )
    await engine.dispose()
    return [dict(row) for row in rows]


def _resolved(messages, tool_call_id: str) -> bool:
    for message in messages:
        for part in getattr(message, "parts", []):
            if (
                getattr(part, "tool_call_id", None) == tool_call_id
                and getattr(part, "part_kind", None) == "tool-return"
            ):
                return True
    return False


def _rename_model(
    *,
    document_id: uuid.UUID,
    title: str,
    tool_call_id: str = TOOL_CALL_ID,
    final_text: str = "Done.",
) -> FunctionModel:
    """Calls `rename_document(document_id, title)` once, with a fixed `tool_call_id`, then --
    once that call's `ToolReturnPart` shows up in history (whether it succeeded or was denied) --
    answers with plain text. Unlike the member flow, an agent-identity call never defers: the tool
    is either denied outright or executed within this same run, so a single `/api/chat` request
    resolves everything."""
    args = {"document_id": str(document_id), "title": title}

    async def call(messages, info: AgentInfo) -> ModelResponse:
        if _resolved(messages, tool_call_id):
            return ModelResponse(parts=[TextPart(final_text)])
        return ModelResponse(
            parts=[ToolCallPart(tool_name="rename_document", args=args, tool_call_id=tool_call_id)]
        )

    async def stream_call(messages, info: AgentInfo):
        if _resolved(messages, tool_call_id):
            yield final_text
        else:
            yield {
                0: DeltaToolCall(
                    name="rename_document", json_args=json.dumps(args), tool_call_id=tool_call_id
                )
            }

    return FunctionModel(call, stream_function=stream_call)


def _headers(identity_id: uuid.UUID) -> dict[str, str]:
    return {"X-Identity-Id": str(identity_id)}


def _chat_path(tenant_id: uuid.UUID) -> str:
    return f"/v1/t/{tenant_id}/api/chat"


def _propose_body(text_: str = "please rename it") -> dict:
    return {
        "id": CONVERSATION_ID,
        "trigger": "submit-message",
        "messages": [{"id": "m1", "role": "user", "parts": [{"type": "text", "text": text_}]}],
    }


@pytest.fixture
def client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_agent_identity_without_a_standing_grant_is_refused_outright(
    environment, client, use_model
):
    """AC1: an agent identity's context calling the writing tool with no active standing grant is
    refused outright -- no pending action is ever created, and no fallback to asking anyone -- and
    an audit record names the denial (`denied_for_lack_of_grant`)."""
    tenant = await seed_tenant(environment, roles=["agent"], via_operator=False)
    identity_id = tenant.identities["agent"]
    await seed_conversation(
        environment,
        tenant_id=tenant.tenant_id,
        identity_id=identity_id,
        conversation_id=CONVERSATION_ID,
    )
    document_id = await seed_document(
        environment, tenant_id=tenant.tenant_id, identity_id=identity_id, title=ORIGINAL_TITLE
    )

    model = _rename_model(document_id=document_id, title=NEW_TITLE)
    use_model(model)

    async with client:
        response = await client.post(
            _chat_path(tenant.tenant_id), json=_propose_body(), headers=_headers(identity_id)
        )
    assert response.status_code == 200, response.text
    # No deferred approval was ever raised for an agent identity -- the denial resolves within
    # this single request.
    assert '"type":"tool-approval-request"' not in response.text

    title = await _document_title(environment.superuser_url, document_id=document_id)
    assert title == ORIGINAL_TITLE  # never executed

    pending_rows = await _pending_action_rows(environment.superuser_url, tenant_id=tenant.tenant_id)
    assert pending_rows == []  # no fallback to asking anyone -- no pending action at all

    events = await _audit_events_for_tenant(environment.superuser_url, tenant_id=tenant.tenant_id)
    assert [e["kind"] for e in events] == ["denied_for_lack_of_grant"]
    assert events[0]["standing_grant_id"] is None
    assert events[0]["pending_action_id"] is None


async def test_agent_identity_with_a_standing_grant_executes_with_no_pending_action(
    environment, client, use_model
):
    """AC2: the same agent-identity context succeeds once an active standing grant for that
    identity and tool exists, executes with no pending action ever created, and the audit record
    for the execution references that grant."""
    tenant = await seed_tenant(environment, roles=["agent", "admin"], via_operator=False)
    identity_id = tenant.identities["agent"]
    agent_membership_id = tenant.memberships["agent"]
    admin_membership_id = tenant.memberships["admin"]
    await seed_conversation(
        environment,
        tenant_id=tenant.tenant_id,
        identity_id=identity_id,
        conversation_id=CONVERSATION_ID,
    )
    document_id = await seed_document(
        environment, tenant_id=tenant.tenant_id, identity_id=identity_id, title=ORIGINAL_TITLE
    )
    grant_id = await _create_standing_grant_via_repository(
        environment,
        tenant_id=tenant.tenant_id,
        agent_membership_id=agent_membership_id,
        granted_by=admin_membership_id,
        tool_name=TOOL_NAME,
    )

    model = _rename_model(document_id=document_id, title=NEW_TITLE)
    use_model(model)

    async with client:
        response = await client.post(
            _chat_path(tenant.tenant_id), json=_propose_body(), headers=_headers(identity_id)
        )
    assert response.status_code == 200, response.text
    assert '"type":"tool-approval-request"' not in response.text

    title = await _document_title(environment.superuser_url, document_id=document_id)
    assert title == NEW_TITLE

    pending_rows = await _pending_action_rows(environment.superuser_url, tenant_id=tenant.tenant_id)
    assert pending_rows == []  # a grant-authorized write never creates a pending action

    events = await _audit_events_for_tenant(environment.superuser_url, tenant_id=tenant.tenant_id)
    assert [e["kind"] for e in events] == ["executed"]
    assert events[0]["standing_grant_id"] == grant_id
    assert events[0]["pending_action_id"] is None


async def test_agent_identity_is_refused_when_the_only_grant_names_a_different_tool(
    environment, client, use_model
):
    """AC3: an agent-identity context is refused when the only active grant it holds names a
    different tool than the one being called."""
    tenant = await seed_tenant(environment, roles=["agent", "admin"], via_operator=False)
    identity_id = tenant.identities["agent"]
    agent_membership_id = tenant.memberships["agent"]
    admin_membership_id = tenant.memberships["admin"]
    await seed_conversation(
        environment,
        tenant_id=tenant.tenant_id,
        identity_id=identity_id,
        conversation_id=CONVERSATION_ID,
    )
    document_id = await seed_document(
        environment, tenant_id=tenant.tenant_id, identity_id=identity_id, title=ORIGINAL_TITLE
    )
    await _create_standing_grant_via_repository(
        environment,
        tenant_id=tenant.tenant_id,
        agent_membership_id=agent_membership_id,
        granted_by=admin_membership_id,
        tool_name="some_other_tool",
    )

    model = _rename_model(document_id=document_id, title=NEW_TITLE)
    use_model(model)

    async with client:
        response = await client.post(
            _chat_path(tenant.tenant_id), json=_propose_body(), headers=_headers(identity_id)
        )
    assert response.status_code == 200, response.text

    title = await _document_title(environment.superuser_url, document_id=document_id)
    assert title == ORIGINAL_TITLE  # never executed

    pending_rows = await _pending_action_rows(environment.superuser_url, tenant_id=tenant.tenant_id)
    assert pending_rows == []

    events = await _audit_events_for_tenant(environment.superuser_url, tenant_id=tenant.tenant_id)
    assert [e["kind"] for e in events] == ["denied_for_lack_of_grant"]
    assert events[0]["standing_grant_id"] is None
