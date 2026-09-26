"""Shared fixtures: context, fake search, TestModel — no real model call, no DB."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime

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
# LITELLM_BASE_URL has no default either (app/config.py, ADR-0009, ai-app-starter#7): the gateway
# is mandatory in every environment, so `Settings()` refuses to construct without it. Set a
# process-wide test default here -- the compose default host ("litellm", allow-listed under
# residency "eu" in config/residency.toml) -- so the suite never depends on a real gateway and
# never makes a network call; a test exercising the "unset" failure constructs
# Settings(litellm_base_url=None, ...) directly, bypassing this env var entirely.
os.environ.setdefault("LITELLM_BASE_URL", "http://litellm:4000")
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
from app.repositories.control import Identity, TenantAuthSettings
from app.repositories.documents import DocumentHit
from app.tenant_record import TenantRecord
from app.token_verifier import set_default_adapter_for_tests

# The suspension timestamp `FakeControlPlaneReads` reports for a tenant its `auth_settings` marks
# suspended (a fixed value: only whether it is set ever matters).
_SUSPENDED_AT = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def ctx() -> RequestContext:
    return RequestContext(
        tenant_id=uuid.uuid4(), identity_id=uuid.uuid4(), roles=frozenset({"member"})
    )


@dataclass
class FakeControlPlaneReads:
    """The one shared fake of `app.token_verifier.ControlPlaneReads` (#100): replaces every
    hand-written per-repository fake (tenant auth settings, memberships, identities) that used to
    be monkeypatched directly onto `app.token_verifier` or `app.tenant_suspension`. Installed as
    the process-wide default via
    `app.token_verifier.set_default_adapter_for_tests` (the `not_suspended` fixture below does
    this for every test by default); a test that needs specific identities, memberships, or
    auth settings installs its own instance the same way, or constructs one and passes it as
    `verify_tenant_token`'s own `adapter=` keyword directly (`tests/test_token_verifier.py`'s
    pattern).

    `identities`: {(issuer, subject): identity_id}.
    `memberships`: {(tenant_id, identity_id): role}.
    `auth_settings`: {tenant_id: (issuer, suspended)} -- a missing key means no control-plane row
    at all, mirroring the real repository's own "no row -> not suspended, fall back to the
    caller's default_issuer" behaviour (ADR-0002).
    `records`: {tenant_id: TenantRecord} (#104). A missing key is the pooled, residency-less
    default record, mirroring the real read's "no row -> pooled, not suspended" -- except that a
    tenant `auth_settings` reports suspended gets a record suspended too, since both real reads
    answer from the same `control.tenants` row and one fake must not contradict itself.
    `explode`: names of `ControlPlaneReads` methods that must never be called at all -- raises
    `AssertionError` if one of them is, for the tests proving an agent-issued token never
    consults the tenant's own auth settings.
    """

    identities: dict[tuple[str, str], uuid.UUID] = field(default_factory=dict)
    memberships: dict[tuple[uuid.UUID, uuid.UUID], str] = field(default_factory=dict)
    auth_settings: dict[uuid.UUID, tuple[str | None, bool]] = field(default_factory=dict)
    records: dict[uuid.UUID, TenantRecord] = field(default_factory=dict)
    explode: frozenset[str] = frozenset()

    def _forbid(self, name: str) -> None:
        if name in self.explode:
            raise AssertionError(f"{name} must not be called")

    async def find_identity_by_issuer_and_subject(
        self, *, issuer: str, subject: str
    ) -> Identity | None:
        self._forbid("find_identity_by_issuer_and_subject")
        identity_id = self.identities.get((issuer, subject))
        if identity_id is None:
            return None
        return Identity(id=identity_id, issuer=issuer, subject=subject)

    async def get_tenant_auth_settings(
        self, *, tenant_id: uuid.UUID, default_issuer: str | None = None
    ) -> TenantAuthSettings | None:
        self._forbid("get_tenant_auth_settings")
        if tenant_id not in self.auth_settings:
            return None
        issuer, suspended = self.auth_settings[tenant_id]
        return TenantAuthSettings(issuer=issuer or default_issuer, suspended=suspended)

    async def get_membership_role(
        self,
        *,
        tenant_id: uuid.UUID,
        identity_id: uuid.UUID,
        tenant_record: TenantRecord | None = None,
    ) -> str | None:
        self._forbid("get_membership_role")
        return self.memberships.get((tenant_id, identity_id))

    async def get_tenant_record(self, *, tenant_id: uuid.UUID) -> TenantRecord:
        self._forbid("get_tenant_record")
        if tenant_id in self.records:
            return self.records[tenant_id]
        _, suspended = self.auth_settings.get(tenant_id, (None, False))
        return TenantRecord(tenant_id=tenant_id, suspended_at=_SUSPENDED_AT if suspended else None)


@pytest.fixture(autouse=True)
def not_suspended():
    """Default fake control-plane-reads adapter (Spec 9 / #69, and #100's adapter seam): no test
    in this file-free suite has a real database, so by default every tenant looks unsuspended --
    mirroring the real repository's own "no control-plane row -> not suspended" default
    (ADR-0002), and every identity/membership lookup returns nothing. Installed once, process-wide,
    via `app.token_verifier.set_default_adapter_for_tests` -- the one seam
    `app.tenant_suspension.ensure_tenant_not_suspended` and `app.token_verifier.verify_tenant_token`
    both fall back to (app/deps.py's dev-headers branch, app/mcp/server.py, app/agents/assistant.py,
    app/api/chat.py all share it). A test that wants a suspended tenant, or specific identities/
    memberships, installs its own `FakeControlPlaneReads` the same way, after this fixture runs.
    """
    set_default_adapter_for_tests(FakeControlPlaneReads())
    yield
    set_default_adapter_for_tests(None)


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
    `app.api.chat.chat` now call in place of the removed, deployment-wide
    `app.llm.get_model()` (ai-app-starter#7).
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
    """TestModel calls every named tool once and answers deterministically.

    Overrides both agents (Spec 5 / #36 split) since a test may exercise either the one-shot
    endpoints or /api/chat without knowing in advance which one it will hit. Restricted to
    `search_documents` (`call_tools=`, rather than the default `'all'`) so a plain functional test
    never drives `chat_assistant`'s writing tool, `rename_document` (ADR-0007, #40) -- that tool's
    own `args_validator` needs a real tenant-bound database session (it writes a pending action),
    which a test using this fixture is not set up to provide. A test that specifically exercises
    the writing tool builds its own `TestModel`/`FunctionModel` against a real database instead
    (see `tests/test_writing_tool_approval_integration.py`). Also patches out per-tenant
    residency-based model resolution (`resolve_to_model`, above) so no real database connection is
    attempted before the override even takes effect.
    """
    tm = TestModel(call_tools=["search_documents"])
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
