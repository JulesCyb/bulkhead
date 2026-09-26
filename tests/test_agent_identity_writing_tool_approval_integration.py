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

CONVERSATION_ID = "conv-agent-1"
TOOL_CALL_ID = "call-rename-agent-1"
ORIGINAL_TITLE = "Original Title"
NEW_TITLE = "Renamed Title"
TOOL_NAME = "rename_document"


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
    """Mirrors `tests/test_writing_tool_approval_integration.py`'s own fixture: app_owner/app
    roles, migrated to head with the real Alembic chain."""
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


async def _seed_standing_grant(
    url: str,
    *,
    tenant_id: uuid.UUID,
    agent_membership_id: uuid.UUID,
    granted_by: uuid.UUID,
    tool_name: str = TOOL_NAME,
) -> uuid.UUID:
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        grant_id = (
            await conn.execute(
                text(
                    "INSERT INTO standing_grants "
                    "(tenant_id, agent_membership_id, tool_name, granted_by) "
                    "VALUES (:tid, :mid, :tool, :granted_by) RETURNING id"
                ),
                {
                    "tid": tenant_id,
                    "mid": agent_membership_id,
                    "tool": tool_name,
                    "granted_by": granted_by,
                },
            )
        ).scalar_one()
    await engine.dispose()
    return grant_id


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
    app_settings, database_urls, client, monkeypatch
):
    """AC1: an agent identity's context calling the writing tool with no active standing grant is
    refused outright -- no pending action is ever created, and no fallback to asking anyone -- and
    an audit record names the denial (`denied_for_lack_of_grant`)."""
    tenant_id = await _seed_tenant(database_urls["superuser"])
    identity_id, _membership_id = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="agent"
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
        response = await client.post(
            _chat_path(tenant_id), json=_propose_body(), headers=_headers(identity_id)
        )
    assert response.status_code == 200, response.text
    # No deferred approval was ever raised for an agent identity -- the denial resolves within
    # this single request.
    assert '"type":"tool-approval-request"' not in response.text

    title = await _document_title(database_urls["superuser"], document_id=document_id)
    assert title == ORIGINAL_TITLE  # never executed

    pending_rows = await _pending_action_rows(database_urls["superuser"], tenant_id=tenant_id)
    assert pending_rows == []  # no fallback to asking anyone -- no pending action at all

    events = await _audit_events_for_tenant(database_urls["superuser"], tenant_id=tenant_id)
    assert [e["kind"] for e in events] == ["denied_for_lack_of_grant"]
    assert events[0]["standing_grant_id"] is None
    assert events[0]["pending_action_id"] is None


async def test_agent_identity_with_a_standing_grant_executes_with_no_pending_action(
    app_settings, database_urls, client, monkeypatch
):
    """AC2: the same agent-identity context succeeds once an active standing grant for that
    identity and tool exists, executes with no pending action ever created, and the audit record
    for the execution references that grant."""
    tenant_id = await _seed_tenant(database_urls["superuser"])
    identity_id, agent_membership_id = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="agent"
    )
    admin_identity_id, admin_membership_id = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="admin"
    )
    await _seed_conversation(
        database_urls["superuser"], tenant_id=tenant_id, identity_id=identity_id
    )
    document_id = await _seed_document(
        database_urls["superuser"], tenant_id=tenant_id, identity_id=identity_id
    )
    grant_id = await _seed_standing_grant(
        database_urls["superuser"],
        tenant_id=tenant_id,
        agent_membership_id=agent_membership_id,
        granted_by=admin_membership_id,
        tool_name=TOOL_NAME,
    )

    model = _rename_model(document_id=document_id, title=NEW_TITLE)
    monkeypatch.setattr(chat_module, "resolve_chat_model", resolve_to_model(model))

    async with client:
        response = await client.post(
            _chat_path(tenant_id), json=_propose_body(), headers=_headers(identity_id)
        )
    assert response.status_code == 200, response.text
    assert '"type":"tool-approval-request"' not in response.text

    title = await _document_title(database_urls["superuser"], document_id=document_id)
    assert title == NEW_TITLE

    pending_rows = await _pending_action_rows(database_urls["superuser"], tenant_id=tenant_id)
    assert pending_rows == []  # a grant-authorized write never creates a pending action

    events = await _audit_events_for_tenant(database_urls["superuser"], tenant_id=tenant_id)
    assert [e["kind"] for e in events] == ["executed"]
    assert events[0]["standing_grant_id"] == grant_id
    assert events[0]["pending_action_id"] is None


async def test_agent_identity_is_refused_when_the_only_grant_names_a_different_tool(
    app_settings, database_urls, client, monkeypatch
):
    """AC3: an agent-identity context is refused when the only active grant it holds names a
    different tool than the one being called."""
    tenant_id = await _seed_tenant(database_urls["superuser"])
    identity_id, agent_membership_id = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="agent"
    )
    _admin_identity_id, admin_membership_id = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="admin"
    )
    await _seed_conversation(
        database_urls["superuser"], tenant_id=tenant_id, identity_id=identity_id
    )
    document_id = await _seed_document(
        database_urls["superuser"], tenant_id=tenant_id, identity_id=identity_id
    )
    await _seed_standing_grant(
        database_urls["superuser"],
        tenant_id=tenant_id,
        agent_membership_id=agent_membership_id,
        granted_by=admin_membership_id,
        tool_name="some_other_tool",
    )

    model = _rename_model(document_id=document_id, title=NEW_TITLE)
    monkeypatch.setattr(chat_module, "resolve_chat_model", resolve_to_model(model))

    async with client:
        response = await client.post(
            _chat_path(tenant_id), json=_propose_body(), headers=_headers(identity_id)
        )
    assert response.status_code == 200, response.text

    title = await _document_title(database_urls["superuser"], document_id=document_id)
    assert title == ORIGINAL_TITLE  # never executed

    pending_rows = await _pending_action_rows(database_urls["superuser"], tenant_id=tenant_id)
    assert pending_rows == []

    events = await _audit_events_for_tenant(database_urls["superuser"], tenant_id=tenant_id)
    assert [e["kind"] for e in events] == ["denied_for_lack_of_grant"]
    assert events[0]["standing_grant_id"] is None
