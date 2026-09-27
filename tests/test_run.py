"""Unit tests of one prepared run (`app.agents.run`, spec A3 / #93, #107): preparation over a
`RequestContext` carrying its tenant record, the two one-shot execution methods, `answer` and
`stream_text`, and the chat method, `chat(adapter)` (#108) -- with the model and the tool
functions injected through `prepare_run`'s own collaborators, never a patched module attribute.

The ASGI routes over the same module are covered in `tests/test_api.py`, `tests/test_chat.py`,
`tests/test_request_limit.py`, `tests/test_content_tracing.py`, and
`tests/test_residency_routing.py`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai import Agent, DeferredToolRequests, RunContext
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

from app import observability
from app.agents.assistant import AssistantDeps, chat_assistant, one_shot_assistant
from app.agents.run import prepare_run
from app.agents.writing_tools import writing_tool
from app.config import Settings
from app.residency import ResidencyUnresolved
from app.run_limits import RunDeadlineExceeded
from app.tenant_record import TenantRecord
from app.tools.approvals import require_approval
from tests.conftest import (
    looping_tool_calls,
    looping_tool_calls_stream,
    make_stalling_model,
    make_stalling_stream_model,
    returning,
)


async def _collect(stream: AsyncIterator[str]) -> str:
    return "".join([delta async for delta in stream])


# --- Preparation ------------------------------------------------------------------------------


async def test_preparation_without_a_tenant_record_fails_closed_before_any_model_call(ctx):
    """A context with no record (a job, a test) has no residency to route a model through: the
    preparation refuses it with `ResidencyUnresolved` -- never a read of its own, never the
    deployment's residency -- and the model resolver is never even asked."""
    asked: list[TenantRecord] = []

    def _resolver(record, *, settings=None):
        asked.append(record)
        return TestModel()

    with pytest.raises(ResidencyUnresolved):
        await prepare_run(ctx, model_resolver=_resolver)
    assert asked == []


async def test_preparation_with_an_unresolved_residency_fails_closed_through_the_real_resolver(
    run_ctx,
):
    """The real model resolver (`app.llm.resolve_tenant_chat_model`): a record with no residency
    recorded fails the preparation closed, before any run exists to call a model."""
    assert run_ctx.tenant_record is not None and run_ctx.tenant_record.residency is None

    with pytest.raises(ResidencyUnresolved):
        await prepare_run(run_ctx)


async def test_preparation_resolves_the_model_from_the_contexts_own_record(run_ctx):
    seen: list[tuple[TenantRecord, Settings | None]] = []
    settings = Settings()

    def _resolver(record, *, settings=None):
        seen.append((record, settings))
        return TestModel()

    await prepare_run(run_ctx, model_resolver=_resolver, settings=settings)

    assert seen == [(run_ctx.tenant_record, settings)]


# --- answer / stream_text ---------------------------------------------------------------------


async def test_answer_reaches_the_tool_with_the_runs_own_context(prepared_run, calls, run_ctx):
    output = await prepared_run.answer("What does the Acme contract say?")

    assert output  # TestModel answers deterministically
    assert calls, "search_documents was not called"
    tenant_id, _query, limit = calls[0]
    assert tenant_id == run_ctx.tenant_id
    assert 1 <= limit <= 20


async def test_stream_text_yields_text_deltas(prepared_run):
    assert await _collect(prepared_run.stream_text("Hello"))


def _function_toolset(agent) -> FunctionToolset:
    for toolset in agent.toolsets:
        if isinstance(toolset, FunctionToolset):
            return toolset
    raise AssertionError(f"{agent} has no function toolset")


def _capturing_model(sink: list[AgentInfo]) -> FunctionModel:
    async def call(messages: list, info: AgentInfo) -> ModelResponse:
        sink.append(info)
        return ModelResponse(parts=[TextPart("ok")])

    async def stream_call(messages: list, info: AgentInfo) -> AsyncIterator[str]:
        sink.append(info)
        yield "ok"

    return FunctionModel(call, stream_function=stream_call)


async def test_answer_and_stream_text_bind_the_reading_only_agent(run_ctx, fake_search):
    """ADR-0007, in one place: the reading-only agent registers no tool with an approval
    validator, and both one-shot execution methods run exactly that agent -- the tool set each
    actually sends to the model is the reading-only agent's, with nothing awaiting approval, and
    the writing-capable chat agent is never reached."""
    reading_tools = _function_toolset(one_shot_assistant).tools
    assert reading_tools, "expected at least one registered tool"
    for name, tool in reading_tools.items():
        assert tool.args_validator is not require_approval, f"{name!r} needs an approval"
        assert not tool.requires_approval, f"reading-only agent must not carry {name!r}"
    writing_tools = {
        name
        for name, tool in _function_toolset(chat_assistant).tools.items()
        if tool.args_validator is require_approval
    }
    assert writing_tools, "the chat agent carries the example writing tool"

    seen: list[AgentInfo] = []
    chat_seen: list[AgentInfo] = []
    with chat_assistant.override(model=_capturing_model(chat_seen)):
        prepared = await prepare_run(
            run_ctx, model_resolver=returning(_capturing_model(seen)), search=fake_search
        )
        await prepared.answer("Hi")
        await _collect(prepared.stream_text("Hi"))

    assert len(seen) == 2
    assert not chat_seen, "a one-shot execution must never reach the chat agent"
    for info in seen:
        names = {tool_def.name for tool_def in info.function_tools}
        assert names == set(reading_tools)
        assert not names & writing_tools
        assert all(tool_def.kind != "unapproved" for tool_def in info.function_tools)


def _chat_adapter(*, agent, conversation_id: str = "conv-1") -> VercelAIAdapter:
    """A Vercel AI SDK adapter over a one-message body, built without a request -- and bound to
    whatever `agent` the caller names, to show that `chat` decides the agent itself."""
    body = {
        "id": conversation_id,
        "trigger": "submit-message",
        "messages": [{"id": "m1", "role": "user", "parts": [{"type": "text", "text": "hi"}]}],
    }
    return VercelAIAdapter(
        agent=agent,
        run_input=VercelAIAdapter.build_run_input(json.dumps(body).encode()),
        sdk_version=6,
    )


async def test_only_chat_binds_the_writing_capable_agent(
    run_ctx, fake_search, fake_history, fake_save
):
    """ADR-0007, the other half: `chat` runs the writing-capable agent -- the tool set it sends to
    the model includes the approval-gated writing tool -- even when handed an adapter built with
    the reading-only agent, since which agent runs is decided by the execution method, never by
    the caller. The route-level mirror is `tests/test_api.py::
    test_chat_endpoint_answers_through_chat_agent`."""
    writing_tools = {
        name
        for name, tool in _function_toolset(chat_assistant).tools.items()
        if tool.args_validator is require_approval
    }
    seen: list[AgentInfo] = []
    prepared = await prepare_run(
        run_ctx,
        conversation_id="conv-1",
        model_resolver=returning(_capturing_model(seen)),
        search=fake_search,
        load_history=fake_history,
        save_run=fake_save,
    )
    response = await prepared.chat(_chat_adapter(agent=one_shot_assistant))
    body = "".join([chunk async for chunk in response.body_iterator])

    assert '"type":"error"' not in body
    assert len(seen) == 1
    names = {tool_def.name for tool_def in seen[0].function_tools}
    assert names == set(_function_toolset(chat_assistant).tools)
    assert writing_tools <= names


async def test_chat_persists_the_run_even_if_the_response_is_never_read(
    run_ctx, fake_search, fake_history
):
    """ADR-0006 (#34), at the module's own seam: the prepared run's new messages are persisted
    once the run completes, even though nothing ever reads a byte of the response `chat`
    returns -- the client decoupling lives in the run module, not in a route. Its ASGI mirror (a
    client disconnecting mid-stream) is `tests/test_chat.py::
    test_persistence_happens_even_when_the_response_is_not_fully_read`."""
    persisted: list[tuple[str, list]] = []
    saved = asyncio.Event()

    async def _save(ctx, conversation_id, messages) -> None:
        persisted.append((conversation_id, messages))
        saved.set()

    prepared = await prepare_run(
        run_ctx,
        conversation_id="conv-1",
        model_resolver=returning(TestModel(call_tools=["search_documents"])),
        search=fake_search,
        load_history=fake_history,
        save_run=_save,
    )
    await prepared.chat(_chat_adapter(agent=chat_assistant))

    await asyncio.wait_for(saved.wait(), timeout=5)
    [(conversation_id, messages)] = persisted
    assert conversation_id == "conv-1"
    assert messages


async def test_chat_refuses_an_adapter_for_a_different_conversation(
    run_ctx, fake_search, fake_history, history_calls
):
    """A run prepared for one conversation never runs against another's adapter: the approval
    scope (`AssistantDeps.conversation_id`) and the history must name the same conversation, so
    a mismatch is refused before any history is loaded."""
    prepared = await prepare_run(
        run_ctx,
        conversation_id="conv-1",
        model_resolver=returning(TestModel()),
        search=fake_search,
        load_history=fake_history,
    )

    with pytest.raises(ValueError, match="different conversation"):
        await prepared.chat(_chat_adapter(agent=chat_assistant, conversation_id="conv-2"))
    assert history_calls == []


async def _prepare_limited(run_ctx, fake_search, model, **settings_overrides):
    settings = Settings(run_tool_calls_limit=2, run_request_limit=50, **settings_overrides)
    return await prepare_run(
        run_ctx, model_resolver=returning(model), search=fake_search, settings=settings
    )


async def test_answer_stops_at_the_run_limit(run_ctx, fake_search):
    prepared = await _prepare_limited(run_ctx, fake_search, FunctionModel(looping_tool_calls))

    with pytest.raises(UsageLimitExceeded, match="tool_calls_limit of 2"):
        await prepared.answer("loop please")


async def test_stream_text_stops_at_the_run_limit(run_ctx, fake_search):
    model = FunctionModel(looping_tool_calls, stream_function=looping_tool_calls_stream)
    prepared = await _prepare_limited(run_ctx, fake_search, model)

    with pytest.raises(UsageLimitExceeded, match="tool_calls_limit of 2"):
        await _collect(prepared.stream_text("loop please"))


async def test_answer_stops_at_the_wall_clock_deadline(run_ctx, fake_search):
    model = FunctionModel(make_stalling_model(seconds=30))
    prepared = await _prepare_limited(run_ctx, fake_search, model, run_deadline_seconds=0.2)

    with pytest.raises(RunDeadlineExceeded):
        await prepared.answer("hang please")


async def test_stream_text_deadline_bounds_consumption_not_only_opening(run_ctx, fake_search):
    """The deadline wraps the full open-and-consume lifecycle inside `stream_text`: a provider
    that stalls after the stream opened still ends the run at its own deadline."""
    model = FunctionModel(
        make_stalling_model(seconds=30), stream_function=make_stalling_stream_model(seconds=30)
    )
    prepared = await _prepare_limited(run_ctx, fake_search, model, run_deadline_seconds=0.2)

    with pytest.raises(RunDeadlineExceeded):
        await _collect(prepared.stream_text("hang please"))


# --- Tracing: capabilities and span attributes from the record and the context ----------------


@pytest.fixture
def eu_spans():
    observability.reset_tracer_providers()
    exporter = InMemorySpanExporter()
    observability.set_tracer_provider_for_residency(
        "eu", observability.build_tracer_provider(exporter, processor_cls=SimpleSpanProcessor)
    )
    yield exporter
    observability.reset_tracer_providers()


@pytest.mark.parametrize("execute", ["answer", "stream_text"])
async def test_every_span_of_either_execution_carries_the_runs_identifiers(
    run_ctx, fake_search, eu_spans, execute
):
    """The tracing selection comes from the record (`eu`, a sink exists), and the span attributes
    from the context -- on every span the run produces, including those created while a stream is
    consumed, since `stream_text` wraps its full lifecycle."""
    record = dataclasses.replace(run_ctx.tenant_record, residency="eu")
    ctx = dataclasses.replace(run_ctx, tenant_record=record)
    prepared = await prepare_run(
        ctx,
        model_resolver=returning(TestModel(call_tools=["search_documents"])),
        search=fake_search,
    )

    if execute == "answer":
        await prepared.answer("Hi")
    else:
        await _collect(prepared.stream_text("Hi"))

    spans = eu_spans.get_finished_spans()
    assert len(spans) > 1
    for span in spans:
        assert span.attributes["tenant_id"] == str(ctx.tenant_id)
        assert span.attributes["identity_id"] == str(ctx.identity_id)
        assert span.attributes["request_id"] == ctx.request_id


# --- The writing_tool decorator (spec A3 / #93, #109) ------------------------------------------


def test_writing_tool_decorator_sets_args_validator_to_require_approval() -> None:
    """`@writing_tool(agent)` alone -- with no other approval wiring at the call site -- produces
    a tool whose `args_validator` is `require_approval` (ADR-0007), the same object
    `rename_document` relies on. Proven on a throwaway agent, not `chat_assistant`, so this test
    says nothing about `rename_document` specifically -- only about what the decorator itself
    does to *any* function it wraps."""
    test_agent: Agent[AssistantDeps, str | DeferredToolRequests] = Agent(
        deps_type=AssistantDeps,
        name="test-writing-tool-agent",
        output_type=[str, DeferredToolRequests],
    )

    @writing_tool(test_agent)
    async def throwaway(ctx: RunContext[AssistantDeps], x: str) -> str | None:
        return x

    tool = _function_toolset(test_agent).tools["throwaway"]
    assert tool.args_validator is require_approval


_APP_DIR = Path(__file__).resolve().parent.parent / "app"
_WRITING_TOOLS_PATH = _APP_DIR / "agents" / "writing_tools.py"
_APPROVALS_PATH = _APP_DIR / "tools" / "approvals.py"


def test_pending_approval_is_read_only_inside_the_writing_tool_decorator() -> None:
    """`ctx.deps.pending_approval` is read back (`approval = ctx.deps.pending_approval`) only by
    `app/agents/writing_tools.py`'s decorator -- never by a route or a tool body directly, which
    would duplicate the wrapper the decorator exists to replace (CLAUDE.md rule 4)."""
    pattern = re.compile(r"=\s*ctx\.deps\.pending_approval\b")
    offenders = []
    for path in sorted(_APP_DIR.rglob("*.py")):
        if path == _WRITING_TOOLS_PATH:
            continue
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            offenders.append(f"{path.relative_to(_APP_DIR)}: {match.group(0)!r}")
    assert offenders == [], offenders


def test_pending_approval_is_set_only_in_the_approvals_module() -> None:
    """`ctx.deps.pending_approval` is set to a real `ApprovalContext` -- the approval hand-off --
    only by `app/tools/approvals.py::require_approval`. The decorator's own clear
    (`ctx.deps.pending_approval = None`) is a different thing (module docstring,
    `app/agents/writing_tools.py`): it never sets a new approval, so it does not match here."""
    pattern = re.compile(r"ctx\.deps\.pending_approval\s*=\s*ApprovalContext\(")
    offenders = []
    for path in sorted(_APP_DIR.rglob("*.py")):
        if path == _APPROVALS_PATH:
            continue
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            offenders.append(f"{path.relative_to(_APP_DIR)}: {match.group(0)!r}")
    assert offenders == [], offenders


# --- No test patches a run's collaborators onto a module (spec A3 / #93, #108) ------------------

_TESTS_DIR = Path(__file__).resolve().parent

# Every way a test used to swap a run's model or tool functions by patching a module attribute
# instead of injecting it through `prepare_run` / `set_run_collaborators_for_tests`: the retired
# chat-route seam, the agent and chat modules' namespaces, and the three tool functions
# `AssistantDeps` already carries as constructor input. The retired seam's name is assembled from
# two halves so that a plain `grep -rn` for it over `tests/` and `app/` (#108's acceptance check)
# finds nothing, this guard included.
_RETIRED_CHAT_SEAM = "resolve_chat" + "_model"
_RETIRED_PATCH_IDIOMS = (
    re.compile(re.escape(_RETIRED_CHAT_SEAM)),
    re.compile(r"setattr\(\s*(assistant_module|chat_module)\b"),
    re.compile(r"setattr\(\s*(assistant_module\.)?document_tools,\s*\"search_documents\""),
    re.compile(
        r"setattr\(\s*(assistant_module\.)?conversation_tools,\s*"
        r"\"(load_conversation_history|save_conversation_run)\""
    ),
)


def test_no_test_patches_a_run_collaborator_onto_a_module() -> None:
    offenders = []
    for path in sorted(_TESTS_DIR.rglob("*.py")):
        if path == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        for idiom in _RETIRED_PATCH_IDIOMS:
            for match in idiom.finditer(text):
                offenders.append(f"{path.relative_to(_TESTS_DIR)}: {match.group(0)!r}")
    assert offenders == [], offenders


def test_the_chat_route_seam_is_gone_from_the_application() -> None:
    app_dir = _TESTS_DIR.parent / "app"
    offenders = [
        str(path.relative_to(app_dir))
        for path in sorted(app_dir.rglob("*.py"))
        if _RETIRED_CHAT_SEAM in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], offenders
