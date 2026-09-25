"""Unit tests for the shared run-limits helper — no ASGI app, no real model call.

HTTP-level behavior (the mapped error at each entry point) is covered in test_api.py and
test_chat.py; this file tests `build_run_limits()`, `run_deadline()`, and `bounded_by_deadline()`
in isolation.
"""

from __future__ import annotations

import time

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from app.config import Settings
from app.run_limits import RunDeadlineExceeded, bounded_by_deadline, build_run_limits, run_deadline
from tests.conftest import looping_tool_calls, make_stalling_model


def test_build_run_limits_reads_configuration():
    limits = build_run_limits(
        Settings(run_request_limit=3, run_tool_calls_limit=4, run_deadline_seconds=12.5)
    )
    assert limits.usage_limits.request_limit == 3
    assert limits.usage_limits.tool_calls_limit == 4
    assert limits.deadline_seconds == 12.5


async def test_run_deadline_ends_a_stalled_run_quickly():
    limits = build_run_limits(Settings(run_deadline_seconds=0.1))
    agent: Agent[None, str] = Agent()

    start = time.monotonic()
    with pytest.raises(RunDeadlineExceeded):
        async with run_deadline(limits):
            await agent.run("hi", model=FunctionModel(make_stalling_model(seconds=10)))
    assert time.monotonic() - start < 5, "the run's own deadline must fire, not the stall itself"


async def test_run_deadline_does_not_affect_a_run_within_its_deadline():
    limits = build_run_limits(Settings(run_deadline_seconds=5.0))
    agent: Agent[None, str] = Agent()

    async with run_deadline(limits):
        result = await agent.run("hi", model=TestModel())
    assert result.output


async def test_bounded_by_deadline_ends_a_stalled_iterator_quickly():
    limits = build_run_limits(Settings(run_deadline_seconds=0.1))

    async def stalled_source():
        yield "first"
        import asyncio

        await asyncio.sleep(10)
        yield "unreachable"  # pragma: no cover

    seen = []
    start = time.monotonic()
    with pytest.raises(RunDeadlineExceeded):
        async for item in bounded_by_deadline(stalled_source(), limits):
            seen.append(item)
    assert seen == ["first"]
    assert time.monotonic() - start < 5


async def test_usage_limits_from_build_run_limits_stop_a_tool_call_loop():
    limits = build_run_limits(Settings(run_tool_calls_limit=2, run_request_limit=50))
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    async def search_documents(query: str) -> str:
        return "ok"

    with pytest.raises(UsageLimitExceeded):
        await agent.run(
            "hi",
            model=FunctionModel(looping_tool_calls),
            usage_limits=limits.usage_limits,
        )
