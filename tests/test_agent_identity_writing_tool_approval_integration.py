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

The tests above authenticate with `AUTH_MODE=dev-headers` (`X-Identity-Id` naming a plain,
directly seeded `agent`-role membership) -- under dev-headers every context resolves as delegation
(`app/context_resolution.py`'s `resolve_dev_headers_context`), so those rows carry
`("agent", "assistant")` as their delegation means, not a credential. The two tests at the bottom
of this file (#117) instead drive the real credential chain end to end --
`AgentIdentityRepository`/`AgentCredentialRepository` create the identity and issue it a
credential, `POST /agent-tokens` exchanges it for a real, signed token
(`AUTH_MODE=jwt`), and that token authenticates the `/api/chat` call -- so `require_approval()`'s
context there really is `RequestContext.acting_through("credential", <public id>)`
(`app.context_resolution.actor_context`), the means ADR-0005 assigns an agent identity acting on
its own credential with no person present.
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

from app.config import Settings, get_settings
from app.context import RequestContext
from app.db.session import tenant_session
from app.main import app
from app.repositories.agent_credentials import AgentCredentialRepository
from app.repositories.agent_identities import AgentIdentityRepository
from app.repositories.standing_grants import StandingGrantRepository
from app.token_verifier import set_default_adapter_for_tests

pgserver = pytest.importorskip("pgserver")

from tests.support import (  # noqa: E402
    cluster,
    environment,
    seed_conversation,
    seed_document,
    seed_tenant,
)

_ = (cluster, environment)

# A real, signed-token settings object for the two credential-means tests at the bottom of this
# file (#117) -- same shape as `tests/test_context_resolution_integration.py`'s own `_jwt_settings`
# helper, kept local since this file needs it for exactly two tests, not every test in it.
_JWT_SECRET = "agent-identity-writing-tool-human-secret-32b"
_AGENT_SECRET = "agent-identity-writing-tool-agent-secret-32b"


def _jwt_settings() -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        auth_mode="jwt",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        default_identity_issuer="seed",
        jwt_verification_key=_JWT_SECRET,
        jwt_algorithm="HS256",
        agent_token_signing_key=_AGENT_SECRET,
        agent_token_ttl_seconds=300,
    )


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
                        "SELECT kind, standing_grant_id, pending_action_id, means_kind, means_id "
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


# --- #117: the same mechanism, authenticated through a real credential (not dev-headers) --------


async def _issue_agent_credential_and_grant(
    admin_ctx: RequestContext, *, admin_membership_id: uuid.UUID, grant: bool
):
    """Creates a real agent identity and issues it a real credential
    (`AgentIdentityRepository`/`AgentCredentialRepository`, exactly as
    `tests/test_agent_identity_end_to_end_integration.py` does), and, when `grant` is true, a
    standing grant for `TOOL_NAME` naming that identity's own membership. Returns
    `(identity_id, issued_credential, grant_id_or_None)`."""
    async with tenant_session(admin_ctx) as session:
        created = await AgentIdentityRepository().create(session, admin_ctx, name="nightly sync")
        issued = await AgentCredentialRepository().create(
            session, admin_ctx, identity_id=created.identity_id, name="nightly sync cred"
        )
        grant_id = None
        if grant:
            standing_grant = await StandingGrantRepository().create(
                session,
                admin_ctx,
                agent_membership_id=created.membership_id,
                tool_name=TOOL_NAME,
                granted_by=admin_membership_id,
            )
            grant_id = standing_grant.id
    return created.identity_id, issued, grant_id


async def _exchange_and_call_chat(
    tenant_id: uuid.UUID, *, public_id: str, secret: str, settings: Settings
) -> httpx.Response:
    """Exchanges a real credential for a real access token through `POST /agent-tokens`
    (`app/api/agent_tokens.py`), then uses that token to authenticate one `POST /api/chat` --
    the same real bearer path `app.deps.get_context`/
    `app.context_resolution.resolve_bearer_context` resolve any other bearer request through,
    never a shortcut construction of `RequestContext`."""
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            token_response = await client.post(
                f"/v1/t/{tenant_id}/agent-tokens", json={"public_id": public_id, "secret": secret}
            )
            assert token_response.status_code == 200, token_response.text
            access_token = token_response.json()["access_token"]
            return await client.post(
                _chat_path(tenant_id),
                json=_propose_body(),
                headers={"Authorization": f"Bearer {access_token}"},
            )
    finally:
        app.dependency_overrides.pop(get_settings, None)


async def test_agent_identity_writing_tool_audit_rows_carry_the_credential_means(
    environment, use_model
):
    """AC (#117): an agent identity authenticated by its own real credential -- not dev-headers'
    `X-Identity-Id` shortcut -- executes a writing tool under a standing grant, and the resulting
    `executed` row carries `means_kind = 'credential'` / `means_id = <the credential's own public
    id>`, alongside (not instead of) the approval means (`standing_grant_id` set,
    `pending_action_id` null)."""
    set_default_adapter_for_tests(None)
    tenant = await seed_tenant(environment, roles=["admin"], via_operator=False)
    admin_ctx = RequestContext(
        tenant_id=tenant.tenant_id,
        identity_id=tenant.identities["admin"],
        roles=frozenset({"admin"}),
    )
    identity_id, issued, grant_id = await _issue_agent_credential_and_grant(
        admin_ctx, admin_membership_id=tenant.memberships["admin"], grant=True
    )
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

    response = await _exchange_and_call_chat(
        tenant.tenant_id, public_id=issued.public_id, secret=issued.secret, settings=_jwt_settings()
    )
    assert response.status_code == 200, response.text

    title = await _document_title(environment.superuser_url, document_id=document_id)
    assert title == NEW_TITLE

    events = await _audit_events_for_tenant(environment.superuser_url, tenant_id=tenant.tenant_id)
    executed = [e for e in events if e["kind"] == "executed"]
    assert len(executed) == 1
    assert executed[0]["means_kind"] == "credential"
    assert executed[0]["means_id"] == issued.public_id
    assert executed[0]["standing_grant_id"] == grant_id
    assert executed[0]["pending_action_id"] is None


async def test_agent_identity_denied_for_lack_of_grant_audit_row_carries_the_credential_means(
    environment, use_model
):
    """AC (#117): the means is known even when no grant exists -- an agent identity refused
    outright for lack of a standing grant still produces a `denied_for_lack_of_grant` row naming
    its own credential as the delegation means."""
    set_default_adapter_for_tests(None)
    tenant = await seed_tenant(environment, roles=["admin"], via_operator=False)
    admin_ctx = RequestContext(
        tenant_id=tenant.tenant_id,
        identity_id=tenant.identities["admin"],
        roles=frozenset({"admin"}),
    )
    identity_id, issued, grant_id = await _issue_agent_credential_and_grant(
        admin_ctx, admin_membership_id=tenant.memberships["admin"], grant=False
    )
    assert grant_id is None
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

    response = await _exchange_and_call_chat(
        tenant.tenant_id, public_id=issued.public_id, secret=issued.secret, settings=_jwt_settings()
    )
    assert response.status_code == 200, response.text

    title = await _document_title(environment.superuser_url, document_id=document_id)
    assert title == ORIGINAL_TITLE  # never executed

    events = await _audit_events_for_tenant(environment.superuser_url, tenant_id=tenant.tenant_id)
    assert [e["kind"] for e in events] == ["denied_for_lack_of_grant"]
    assert events[0]["means_kind"] == "credential"
    assert events[0]["means_id"] == issued.public_id
    assert events[0]["standing_grant_id"] is None
    assert events[0]["pending_action_id"] is None
