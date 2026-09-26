"""Unit tests for the agent-run entry points' own suspension check (Spec 9 / #69, ADR-0010):
`run_assistant` and `stream_assistant` (`app/agents/assistant.py`) reject a suspended tenant
before any tool runs, independently of whatever HTTP-layer check (`app/deps.py`) built their
`RequestContext` -- exercised directly here, with no ASGI request and no real database, unlike
`tests/test_api.py`'s HTTP-level coverage of the same two endpoints.
"""

from __future__ import annotations

import pytest

from app.agents.assistant import AssistantDeps, run_assistant, stream_assistant
from app.tenant_suspension import TenantSuspendedError
from app.token_verifier import set_default_adapter_for_tests
from tests.conftest import FakeControlPlaneReads


def _suspend(ctx) -> None:
    set_default_adapter_for_tests(
        FakeControlPlaneReads(auth_settings={ctx.tenant_id: (None, True)})
    )


async def test_run_assistant_rejects_a_suspended_tenant_before_any_tool_runs(ctx):
    _suspend(ctx)

    async def _boom(ctx, query, limit):
        pytest.fail("no tool must run for a suspended tenant")

    with pytest.raises(TenantSuspendedError):
        await run_assistant("hi", AssistantDeps(ctx=ctx, search=_boom))


async def test_stream_assistant_rejects_a_suspended_tenant_before_opening_the_stream(ctx):
    _suspend(ctx)

    async def _boom(ctx, query, limit):
        pytest.fail("no tool must run for a suspended tenant")

    with pytest.raises(TenantSuspendedError):
        async with stream_assistant("hi", AssistantDeps(ctx=ctx, search=_boom)):
            pytest.fail("the stream must never open for a suspended tenant")
