"""Unit tests of one prepared run (`app.agents.run`, spec A3 / #93, #107): preparation over a
`RequestContext` carrying its tenant record, and the two one-shot execution methods, `answer` and
`stream_text` -- with the model and the tool functions injected through `prepare_run`'s own
collaborators, never a patched module attribute.

The ASGI routes over the same module are covered in `tests/test_api.py`,
`tests/test_request_limit.py`, `tests/test_content_tracing.py`, and
`tests/test_residency_routing.py`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets.function import FunctionToolset

from app import observability
from app.agents.assistant import chat_assistant, one_shot_assistant
from app.agents.run import prepare_run
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
