"""ASGI-seam tests for the writing-tool approval round-trip (ADR-0007, Spec 5 / #40), against
PostgreSQL + pgvector (`pgserver`, via `uv sync --group dbtest`). Pattern:
`tests/test_standing_grants_integration.py` / `tests/test_approval_audit_integration.py`, driven
end to end through `POST /v1/t/{tenant_id}/api/chat` with a `FunctionModel` standing in for the
model -- no real model call.

`rename_document` (`app/agents/assistant.py`, `app/tools/documents.py`) is the example writing
tool ADR-0007 and this ticket ask for; `require_approval`/`resolve_incoming_decisions`
(`app/tools/approvals.py`) are the machinery under test. These tests prove what a unit test on
`PendingActionRepository`/`ApprovalAuditRepository` alone cannot: that the whole mechanism holds
together across two real HTTP requests against a real database, in the exact order a client would
see it.

The model is swapped in the same way `tests/test_chat.py` already does --
`monkeypatch.setattr(chat_module, "resolve_chat_model", resolve_to_model(...))` -- rather than
`chat_assistant.override(...)`: the chat endpoint always resolves an explicit model
(`app.agents.assistant.resolve_chat_model`, per-tenant/residency routed, Spec 8 / #61) and passes
it into `adapter.run_stream(...)`, which shadows an agent-level `.override(model=...)` entirely.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.api import chat as chat_module
from app.db.guard import ROLE_STATEMENT_TIMEOUT_MS
from app.main import app
from tests.conftest import resolve_to_model

pgserver = pytest.importorskip("pgserver")

CONVERSATION_ID = "conv-1"
TOOL_CALL_ID = "call-rename-1"
ORIGINAL_TITLE = "Original Title"
NEW_TITLE = "Renamed Title"


def _psql(server, command: str) -> None:
    """`server.psql` without a shell: pgserver's own version breaks on paths with spaces."""
    from pgserver.postgres_server import POSTGRES_BIN_PATH

    subprocess.run(
        [str(POSTGRES_BIN_PATH / "psql"), server.get_uri()],
        input=command.encode(),
        check=True,
        capture_output=True,
    )


@pytest.fixture(scope="module")
def database_urls():
    """Mirrors `tests/test_standing_grants_integration.py`'s own fixture: app_owner/app roles,
    migrated to head with the real Alembic chain (including #40's own
    0038_pending_action_tool_call_id)."""
    pgdata = tempfile.mkdtemp(prefix="pgdata-")
    server = pgserver.get_server(pgdata, cleanup_mode="delete")
    sockdir = parse_qs(urlparse(server.get_uri()).query)["host"][0]
    _psql(
        server,
        "CREATE EXTENSION IF NOT EXISTS vector; "
        "CREATE ROLE app_owner LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE; "
        "ALTER SCHEMA public OWNER TO app_owner; "
        "GRANT CREATE ON DATABASE postgres TO app_owner; "
        "CREATE ROLE app LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE; "
        "GRANT USAGE ON SCHEMA public TO app; "
        f"ALTER ROLE app SET statement_timeout = '{ROLE_STATEMENT_TIMEOUT_MS}ms';",
    )
    urls = {
        "migrations": f"postgresql+asyncpg://app_owner@/postgres?host={sockdir}",
        "app": f"postgresql+asyncpg://app@/postgres?host={sockdir}",
        "superuser": f"postgresql+asyncpg://postgres@/postgres?host={sockdir}",
    }
    env = {**os.environ, "DATABASE_URL_MIGRATIONS": urls["migrations"], "DATABASE_URL": urls["app"]}
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"], check=True, env=env, timeout=120
    )
    yield urls
    server.cleanup()


@pytest.fixture
def app_settings(database_urls, monkeypatch):
    from app import config
    from app.db import session as db_session

    monkeypatch.setenv("DATABASE_URL", database_urls["app"])
    monkeypatch.setenv("DATABASE_URL_MIGRATIONS", database_urls["migrations"])
    monkeypatch.setenv("PENDING_ACTION_EXPIRY_SECONDS", "300")
    config.get_settings.cache_clear()
    db_session._engine = None
    db_session._session_factory = None
    yield
    config.get_settings.cache_clear()
    db_session._engine = None
    db_session._session_factory = None


async def _seed_tenant(url: str) -> uuid.UUID:
    engine = create_async_engine(url)
    tenant_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, 'Acme')"), {"id": tenant_id}
        )
    await engine.dispose()
    return tenant_id


async def _seed_membership(
    url: str, *, tenant_id: uuid.UUID, role: str
) -> tuple[uuid.UUID, uuid.UUID]:
    """A global identity plus its membership of `role` in `tenant_id`. Returns
    (identity_id, membership_id)."""
    engine = create_async_engine(url)
    identity_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO control.identities (id, issuer, subject) VALUES (:id, 'seed', :sub)"),
            {"id": identity_id, "sub": str(identity_id)},
        )
        membership_id = (
            await conn.execute(
                text(
                    "INSERT INTO memberships (tenant_id, identity_id, role) "
                    "VALUES (:tid, :iid, :role) RETURNING id"
                ),
                {"tid": tenant_id, "iid": identity_id, "role": role},
            )
        ).scalar_one()
    await engine.dispose()
    return identity_id, membership_id


async def _seed_conversation(url: str, *, tenant_id: uuid.UUID, identity_id: uuid.UUID) -> None:
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO conversations (tenant_id, conversation_id, created_by) "
                "VALUES (:tid, :cid, :creator)"
            ),
            {"tid": tenant_id, "cid": CONVERSATION_ID, "creator": identity_id},
        )
    await engine.dispose()


async def _seed_document(
    url: str, *, tenant_id: uuid.UUID, identity_id: uuid.UUID, title: str = ORIGINAL_TITLE
) -> uuid.UUID:
    engine = create_async_engine(url)
    document_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO documents (id, tenant_id, title, content, created_by, updated_by) "
                "VALUES (:id, :tid, :title, 'content', :who, :who)"
            ),
            {"id": document_id, "tid": tenant_id, "title": title, "who": identity_id},
        )
    await engine.dispose()
    return document_id


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


async def _pending_action_rows(url: str, *, tenant_id: uuid.UUID) -> list[dict]:
    engine = create_async_engine(url)
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    text(
                        "SELECT id, status, tool_call_id, expires_at FROM pending_actions "
                        "WHERE tenant_id = :tid"
                    ),
                    {"tid": tenant_id},
                )
            )
            .mappings()
            .all()
        )
    await engine.dispose()
    return [dict(row) for row in rows]


async def _set_membership_role(url: str, *, membership_id: uuid.UUID, role: str) -> None:
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE memberships SET role = :role WHERE id = :id"),
            {"role": role, "id": membership_id},
        )
    await engine.dispose()


async def _set_pending_action_expiry_in_the_past(url: str, *, tenant_id: uuid.UUID) -> None:
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE pending_actions SET expires_at = now() - interval '1 hour' "
                "WHERE tenant_id = :tid"
            ),
            {"tid": tenant_id},
        )
    await engine.dispose()


async def _tamper_stored_tool_call_args(
    url: str, *, tenant_id: uuid.UUID, tool_call_id: str, new_args: dict
) -> None:
    """Rewrites the `args` of the persisted `ToolCallPart` matching `tool_call_id`, directly in
    the `messages` table -- simulates a tampered call the same way an attacker who could touch
    stored history (or a forged resend) would: the *stored* call the resumed run resolves against
    now differs from the one `PendingActionRepository.create()` hashed when the write was first
    proposed. A real client cannot change a deferred call's own arguments through the approval
    protocol itself (they are fixed in history from the first run, not re-supplied by the model on
    resume) -- this is the direct route to the same tampered-argument scenario
    `PendingActionRepository.verify()`'s hash check exists to catch."""
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        rows = (
            (
                await conn.execute(
                    text("SELECT id, payload FROM messages WHERE tenant_id = :tid"),
                    {"tid": tenant_id},
                )
            )
            .mappings()
            .all()
        )
        for row in rows:
            payload = row["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            changed = False
            for part in payload.get("parts", []):
                if (
                    part.get("part_kind") == "tool-call"
                    and part.get("tool_call_id") == tool_call_id
                ):
                    part["args"] = new_args
                    changed = True
            if changed:
                await conn.execute(
                    text("UPDATE messages SET payload = :payload WHERE id = :id"),
                    {"payload": json.dumps(payload), "id": row["id"]},
                )
    await engine.dispose()


async def _audit_kinds_for_tenant(url: str, *, tenant_id: uuid.UUID) -> list[str]:
    engine = create_async_engine(url)
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    text(
                        "SELECT kind FROM approval_audit_events WHERE tenant_id = :tid ORDER BY seq"
                    ),
                    {"tid": tenant_id},
                )
            )
            .scalars()
            .all()
        )
    await engine.dispose()
    return list(rows)


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
    """Calls `rename_document(document_id, title)` once, with a fixed `tool_call_id` the test
    controls, then -- once that call's `ToolReturnPart` shows up in history (approved, denied, or
    failed; `build_tool_return_part` produces one for all three) -- answers with plain text."""
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


def _resume_body(
    *,
    tool_call_id: str,
    document_id: uuid.UUID,
    title: str,
    approved: bool,
    reason: str | None = None,
) -> dict:
    approval: dict = {"id": f"approval-{tool_call_id}", "approved": approved}
    if reason is not None:
        approval["reason"] = reason
    return {
        "id": CONVERSATION_ID,
        "trigger": "submit-message",
        "messages": [
            {
                "id": "m2",
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool-rename_document",
                        "toolCallId": tool_call_id,
                        "state": "approval-responded",
                        "input": {"document_id": str(document_id), "title": title},
                        "approval": approval,
                    }
                ],
            }
        ],
    }


async def _propose(client: httpx.AsyncClient, tenant_id: uuid.UUID, identity_id: uuid.UUID) -> str:
    response = await client.post(
        _chat_path(tenant_id), json=_propose_body(), headers=_headers(identity_id)
    )
    assert response.status_code == 200, response.text
    assert '"type":"tool-approval-request"' in response.text
    return response.text


async def _resume(
    client: httpx.AsyncClient,
    tenant_id: uuid.UUID,
    identity_id: uuid.UUID,
    *,
    document_id: uuid.UUID,
    title: str,
    approved: bool,
    reason: str | None = None,
    tool_call_id: str = TOOL_CALL_ID,
) -> httpx.Response:
    response = await client.post(
        _chat_path(tenant_id),
        json=_resume_body(
            tool_call_id=tool_call_id,
            document_id=document_id,
            title=title,
            approved=approved,
            reason=reason,
        ),
        headers=_headers(identity_id),
    )
    assert response.status_code == 200, response.text
    return response


@pytest.fixture
def client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_pending_action_exists_before_the_deferred_approval_reaches_the_client(
    app_settings, database_urls, client, monkeypatch
):
    """AC1: the pending action behind a deferred approval is already committed to storage before
    the response naming it reaches the client -- proven by streaming the response and checking
    storage the moment the `tool-approval-request` chunk is seen, not merely after the whole
    response has been read (ordering, not just eventual presence). `handle_run_result` only emits
    that chunk once the whole run has completed (`pydantic_ai.ui.vercel_ai._event_stream`), and the
    validator's own commit happens strictly earlier in the same coroutine chain -- this test proves
    that holds through the real ASGI/streaming plumbing, not just by inspection of the source."""
    tenant_id = await _seed_tenant(database_urls["superuser"])
    identity_id, _ = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="member"
    )
    await _seed_conversation(
        database_urls["superuser"], tenant_id=tenant_id, identity_id=identity_id
    )
    document_id = await _seed_document(
        database_urls["superuser"], tenant_id=tenant_id, identity_id=identity_id
    )

    model = _rename_model(document_id=document_id, title=NEW_TITLE)
    monkeypatch.setattr(chat_module, "resolve_chat_model", resolve_to_model(model))

    async with client:
        seen_chunk = False
        async with client.stream(
            "POST", _chat_path(tenant_id), json=_propose_body(), headers=_headers(identity_id)
        ) as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                if '"type":"tool-approval-request"' in line:
                    seen_chunk = True
                    # The row must already be there -- checked from a second, independent
                    # connection, right after seeing the chunk that announces it.
                    rows = await _pending_action_rows(
                        database_urls["superuser"], tenant_id=tenant_id
                    )
                    assert len(rows) == 1
                    assert rows[0]["status"] == "pending"
                    assert rows[0]["tool_call_id"] == TOOL_CALL_ID
                    break
        assert seen_chunk, "never saw the tool-approval-request chunk"


async def test_approving_executes_exactly_once_and_response_reflects_the_change(
    app_settings, database_urls, client, monkeypatch
):
    """AC2 (approve half): resuming with an approval executes the tool exactly once, and the
    document's title is actually changed through the repository layer."""
    tenant_id = await _seed_tenant(database_urls["superuser"])
    identity_id, _ = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="member"
    )
    await _seed_conversation(
        database_urls["superuser"], tenant_id=tenant_id, identity_id=identity_id
    )
    document_id = await _seed_document(
        database_urls["superuser"], tenant_id=tenant_id, identity_id=identity_id
    )

    model = _rename_model(document_id=document_id, title=NEW_TITLE)
    monkeypatch.setattr(chat_module, "resolve_chat_model", resolve_to_model(model))

    async with client:
        await _propose(client, tenant_id, identity_id)
        resumed = await _resume(
            client, tenant_id, identity_id, document_id=document_id, title=NEW_TITLE, approved=True
        )

    assert "Renamed document" in resumed.text
    title = await _document_title(database_urls["superuser"], document_id=document_id)
    assert title == NEW_TITLE

    kinds = await _audit_kinds_for_tenant(database_urls["superuser"], tenant_id=tenant_id)
    assert kinds == ["requested", "approved", "executed"]


async def test_refusing_never_executes_and_the_conversation_continues(
    app_settings, database_urls, client, monkeypatch
):
    """AC2 (refuse half): resuming with a refusal leaves the tool never executed, and the
    conversation still produces a normal reply."""
    tenant_id = await _seed_tenant(database_urls["superuser"])
    identity_id, _ = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="member"
    )
    await _seed_conversation(
        database_urls["superuser"], tenant_id=tenant_id, identity_id=identity_id
    )
    document_id = await _seed_document(
        database_urls["superuser"], tenant_id=tenant_id, identity_id=identity_id
    )

    model = _rename_model(document_id=document_id, title=NEW_TITLE)
    monkeypatch.setattr(chat_module, "resolve_chat_model", resolve_to_model(model))

    async with client:
        await _propose(client, tenant_id, identity_id)
        resumed = await _resume(
            client,
            tenant_id,
            identity_id,
            document_id=document_id,
            title=NEW_TITLE,
            approved=False,
            reason="not now",
        )

    assert resumed.status_code == 200
    assert '"type":"error"' not in resumed.text

    title = await _document_title(database_urls["superuser"], document_id=document_id)
    assert title == ORIGINAL_TITLE  # never executed

    kinds = await _audit_kinds_for_tenant(database_urls["superuser"], tenant_id=tenant_id)
    assert kinds == ["requested", "refused"]


async def test_tampered_arguments_on_approval_are_refused_nothing_executed(
    app_settings, database_urls, client, monkeypatch
):
    """AC3: an approval whose arguments differ from what was originally proposed is refused, with
    nothing executed. A real client cannot change a deferred call's own arguments through the
    approval protocol (they are fixed, from the first run's own `ToolCallPart`, not re-supplied by
    the model on resume) -- this simulates the equivalent tampering directly on the stored,
    trusted history, and proves `PendingActionRepository.verify()`'s hash check catches it."""
    tenant_id = await _seed_tenant(database_urls["superuser"])
    identity_id, _ = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="member"
    )
    await _seed_conversation(
        database_urls["superuser"], tenant_id=tenant_id, identity_id=identity_id
    )
    document_id = await _seed_document(
        database_urls["superuser"], tenant_id=tenant_id, identity_id=identity_id
    )

    model = _rename_model(document_id=document_id, title=NEW_TITLE)
    monkeypatch.setattr(chat_module, "resolve_chat_model", resolve_to_model(model))

    async with client:
        await _propose(client, tenant_id, identity_id)

        await _tamper_stored_tool_call_args(
            database_urls["superuser"],
            tenant_id=tenant_id,
            tool_call_id=TOOL_CALL_ID,
            new_args={"document_id": str(document_id), "title": "TAMPERED TITLE"},
        )

        resumed = await _resume(
            client,
            tenant_id,
            identity_id,
            document_id=document_id,
            title=NEW_TITLE,
            approved=True,
        )

    assert resumed.status_code == 200
    title = await _document_title(database_urls["superuser"], document_id=document_id)
    assert title == ORIGINAL_TITLE  # nothing executed

    kinds = await _audit_kinds_for_tenant(database_urls["superuser"], tenant_id=tenant_id)
    assert kinds == ["requested", "approved", "failed_to_execute"]


async def test_role_downgraded_between_request_and_resume_is_refused(
    app_settings, database_urls, client, monkeypatch
):
    """AC4: a membership whose role is downgraded between the initial request and the resumed
    approval causes the tool's own execution-time check to refuse -- distinct from, and in
    addition to, the check already made when the write was first proposed (the proposal itself
    succeeded while the membership was still `member`)."""
    tenant_id = await _seed_tenant(database_urls["superuser"])
    identity_id, membership_id = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="member"
    )
    await _seed_conversation(
        database_urls["superuser"], tenant_id=tenant_id, identity_id=identity_id
    )
    document_id = await _seed_document(
        database_urls["superuser"], tenant_id=tenant_id, identity_id=identity_id
    )

    model = _rename_model(document_id=document_id, title=NEW_TITLE)
    monkeypatch.setattr(chat_module, "resolve_chat_model", resolve_to_model(model))

    async with client:
        await _propose(client, tenant_id, identity_id)

        await _set_membership_role(
            database_urls["superuser"], membership_id=membership_id, role="support"
        )

        resumed = await _resume(
            client, tenant_id, identity_id, document_id=document_id, title=NEW_TITLE, approved=True
        )

    assert resumed.status_code == 200
    title = await _document_title(database_urls["superuser"], document_id=document_id)
    assert title == ORIGINAL_TITLE  # nothing executed

    kinds = await _audit_kinds_for_tenant(database_urls["superuser"], tenant_id=tenant_id)
    assert kinds == ["requested", "approved", "failed_to_execute"]


async def test_expired_approval_is_refused_and_marked_expired(
    app_settings, database_urls, client, monkeypatch
):
    """AC5: an approval answered after the configured expiry window is refused, and the resulting
    audit record marks it `expired` -- exercised against the real wall clock (the pending action's
    `expires_at` is moved into the past directly, the same fail-closed check
    `PendingActionRepository.verify()` already uses for `tests/test_rls_integration.py`)."""
    tenant_id = await _seed_tenant(database_urls["superuser"])
    identity_id, _ = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="member"
    )
    await _seed_conversation(
        database_urls["superuser"], tenant_id=tenant_id, identity_id=identity_id
    )
    document_id = await _seed_document(
        database_urls["superuser"], tenant_id=tenant_id, identity_id=identity_id
    )

    model = _rename_model(document_id=document_id, title=NEW_TITLE)
    monkeypatch.setattr(chat_module, "resolve_chat_model", resolve_to_model(model))

    async with client:
        await _propose(client, tenant_id, identity_id)

        await _set_pending_action_expiry_in_the_past(
            database_urls["superuser"], tenant_id=tenant_id
        )

        resumed = await _resume(
            client, tenant_id, identity_id, document_id=document_id, title=NEW_TITLE, approved=True
        )

    assert resumed.status_code == 200
    title = await _document_title(database_urls["superuser"], document_id=document_id)
    assert title == ORIGINAL_TITLE

    kinds = await _audit_kinds_for_tenant(database_urls["superuser"], tenant_id=tenant_id)
    assert kinds == ["requested", "approved", "expired"]


async def test_writing_tool_goes_through_the_shared_repository_layer(
    app_settings, database_urls, client, monkeypatch
):
    """AC7: the example writing tool's data access is the same repository layer a reading tool
    uses -- proven the same way `search_documents` is: the change is visible directly at the
    database layer, through `DocumentRepository.rename`, not a bespoke connection of the tool's
    own. `updated_by` is refreshed by the same `documents_set_update_audit` trigger (migration
    0010) every other write to `documents` goes through -- a bespoke connection bypassing the
    repository would not trigger it."""
    tenant_id = await _seed_tenant(database_urls["superuser"])
    identity_id, _ = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="admin"
    )
    await _seed_conversation(
        database_urls["superuser"], tenant_id=tenant_id, identity_id=identity_id
    )
    document_id = await _seed_document(
        database_urls["superuser"], tenant_id=tenant_id, identity_id=identity_id
    )

    model = _rename_model(document_id=document_id, title=NEW_TITLE)
    monkeypatch.setattr(chat_module, "resolve_chat_model", resolve_to_model(model))

    async with client:
        await _propose(client, tenant_id, identity_id)
        await _resume(
            client, tenant_id, identity_id, document_id=document_id, title=NEW_TITLE, approved=True
        )

    engine = create_async_engine(database_urls["superuser"])
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text("SELECT title, updated_by FROM documents WHERE id = :id"), {"id": document_id}
            )
        ).one()
    await engine.dispose()
    assert row.title == NEW_TITLE
    assert row.updated_by == identity_id
