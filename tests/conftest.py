"""Shared fixtures: context, fake search, TestModel — no real model call, no DB."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCalls
from pydantic_ai.models.function import DeltaToolCall as _DeltaToolCall
from pydantic_ai.models.test import TestModel

# EMBEDDING_PROVIDER/EMBEDDING_MODEL have no default (app/config.py) — Settings() refuses to
# construct without them. Set process-wide test defaults here, at collection time, so tests that
# don't care about embedding config (most of the suite) don't each have to supply it. Tests that
# exercise the "unset" failure construct Settings(embedding_provider=None, ...) directly, which
# bypasses these env vars entirely.
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-3-small")
# ENVIRONMENT/AUTH_MODE have no default either (issue #14 / ADR-0011): a `.env` copied and left
# unedited must fail to start rather than silently choosing `dev`/`dev-headers`. Same pattern as
# above — tests that exercise the fail-closed default itself delete these from the environment
# and pass `_env_file=None` directly.
os.environ.setdefault("ENVIRONMENT", "dev")
os.environ.setdefault("AUTH_MODE", "dev-headers")

from app.agents import assistant as assistant_module
from app.agents.assistant import AssistantDeps, chat_assistant, one_shot_assistant
from app.api import chat as chat_module
from app.context import RequestContext
from app.repositories.documents import DocumentHit


@pytest.fixture
def ctx() -> RequestContext:
    return RequestContext(
        tenant_id=uuid.uuid4(), identity_id=uuid.uuid4(), roles=frozenset({"member"})
    )


@pytest.fixture(autouse=True)
def not_suspended(monkeypatch):
    """Default fake for `app.tenant_suspension.ensure_tenant_not_suspended`'s one control-plane
    read (Spec 9 / #69): no test in this file-free suite has a real database, so by default every
    tenant looks unsuspended -- mirroring the real repository's own "no control-plane row -> not
    suspended" default (ADR-0002). Every caller (app/deps.py's dev-headers branch,
    app/mcp/server.py, app/agents/assistant.py, app/api/chat.py) shares this one seam, so patching
    it here once is enough for the whole ASGI/MCP/agent-run test suite; a test that wants a
    suspended tenant re-patches `TenantAuthSettingsRepository.get` (or `control_session`) itself,
    after this fixture runs, to report one before making its request.
    """
    import app.tenant_suspension as tenant_suspension_module

    @asynccontextmanager
    async def _fake_control_session():
        yield None

    class _FakeTenantAuthSettingsRepository:
        async def get(self, session, *, tenant_id, default_issuer=None):
            return None

    monkeypatch.setattr(tenant_suspension_module, "control_session", _fake_control_session)
    monkeypatch.setattr(
        tenant_suspension_module,
        "TenantAuthSettingsRepository",
        _FakeTenantAuthSettingsRepository,
    )


@pytest.fixture
def calls() -> list[tuple[uuid.UUID, str, int]]:
    return []


@pytest.fixture
def contexts() -> list[RequestContext]:
    """Every RequestContext a fake tool call actually received, in order."""
    return []


@pytest.fixture
def fake_search(calls, contexts):
    async def _search(ctx: RequestContext, query: str, limit: int) -> list[DocumentHit]:
        calls.append((ctx.tenant_id, query, limit))
        contexts.append(ctx)
        return [DocumentHit(id=uuid.uuid4(), title="Acme contract", snippet="…", score=0.9)]

    return _search


@pytest.fixture
def history_calls() -> list[tuple[uuid.UUID, str]]:
    """Every (tenant_id, conversation_id) a fake `load_history` was actually asked for."""
    return []


@pytest.fixture
def fake_history(history_calls):
    """No stored history, by default (ADR-0006, #33) — a test that cares about a specific
    stored history writes its own fake instead of using this one."""

    async def _load(ctx: RequestContext, conversation_id: str) -> list[ModelMessage]:
        history_calls.append((ctx.tenant_id, conversation_id))
        return []

    return _load


@pytest.fixture
def save_calls() -> list[tuple[uuid.UUID, str, list[ModelMessage]]]:
    """Every (tenant_id, conversation_id, messages) a fake `save_run` was actually asked to
    persist -- ADR-0006, #34."""
    return []


@pytest.fixture
def fake_save(save_calls):
    """Records the run's persisted messages in-memory instead of touching a database — a test
    that cares about a specific store (e.g. seeing an earlier run's reply on a second request)
    writes its own fake keyed by conversation id instead of using this one."""

    async def _save(
        ctx: RequestContext, conversation_id: str, messages: list[ModelMessage]
    ) -> None:
        save_calls.append((ctx.tenant_id, conversation_id, messages))

    return _save


@pytest.fixture
def deps(ctx, fake_search, fake_history, fake_save) -> AssistantDeps:
    return AssistantDeps(ctx=ctx, search=fake_search, load_history=fake_history, save_run=fake_save)


def resolve_to_model(model):
    """Wraps `model` as a fake `resolve_chat_model` (Spec 8 / #61): the per-tenant, residency-
    routed model resolver `app.agents.assistant.run_assistant`/`stream_assistant` and
    `app.api.chat.chat` now call in place of the old, deployment-wide `app.llm.get_model()`.
    Ignores `deps` entirely and always returns `model` — the test seam every ASGI test in this
    suite uses to inject a `TestModel`/`FunctionModel` without a real database or gateway
    credential file on disk. Patch both `app.agents.assistant.resolve_chat_model` (used by the
    one-shot run/stream entry points) and `app.api.chat.resolve_chat_model` (its own
    `from ... import` binding, a separate name to patch) to cover every entry point a test drives.
    """

    async def _resolve(deps):
        return model

    return _resolve


@pytest.fixture
def test_model(monkeypatch):
    """TestModel calls every tool once and answers deterministically.

    Overrides both agents (Spec 5 / #36 split) since a test may exercise either the one-shot
    endpoints or /api/chat without knowing in advance which one it will hit. Also patches out
    per-tenant residency-based model resolution (`resolve_to_model`, above) so no real database
    connection is attempted before the override even takes effect.
    """
    tm = TestModel()
    resolver = resolve_to_model(tm)
    monkeypatch.setattr(assistant_module, "resolve_chat_model", resolver)
    monkeypatch.setattr(chat_module, "resolve_chat_model", resolver)
    with one_shot_assistant.override(model=tm), chat_assistant.override(model=tm):
        yield tm


def looping_tool_calls(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """An engineered run: calls `search_documents` again on every step, forever — the kind of
    loop a poisoned document can provoke. Used to drive an agent run past its tool-call ceiling.
    """
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name="search_documents",
                args={"query": "again"},
                tool_call_id=f"call-{len(messages)}",
            )
        ]
    )


async def looping_tool_calls_stream(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[DeltaToolCalls]:
    """Streamed counterpart of `looping_tool_calls`, for `FunctionModel(stream_function=...)`."""
    yield {0: _DeltaToolCall(name="search_documents", json_args='{"query": "again"}')}


def make_stalling_model(seconds: float = 10.0):
    """A `FunctionModel` function that never returns within any run's configured deadline —
    simulates a stalled provider without a real network call or a real sleep beyond `seconds`
    (the caller picks a short one; the run's own deadline is expected to fire first).
    """

    async def _stall(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        await asyncio.sleep(seconds)
        return ModelResponse(parts=[TextPart(content="unreachable")])  # pragma: no cover

    return _stall


def make_stalling_stream_model(seconds: float = 10.0):
    """Streamed counterpart of `make_stalling_model`, for `FunctionModel(stream_function=...)`."""

    async def _stall(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        await asyncio.sleep(seconds)
        yield "unreachable"  # pragma: no cover

    return _stall
