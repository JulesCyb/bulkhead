"""Unit tests for the agent-run entry points' own suspension check (Spec 9 / #69, ADR-0010):
`run_assistant` and `stream_assistant` (`app/agents/assistant.py`) reject a suspended tenant
before any tool runs, independently of whatever HTTP-layer check (`app/deps.py`) built their
`RequestContext` -- exercised directly here, with no ASGI request and no real database, unlike
`tests/test_api.py`'s HTTP-level coverage of the same two endpoints.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

import app.tenant_suspension as tenant_suspension_module
from app.agents.assistant import AssistantDeps, run_assistant, stream_assistant
from app.tenant_suspension import TenantSuspendedError


def _suspend(monkeypatch) -> None:
    @asynccontextmanager
    async def _fake_control_session():
        yield None

    class _FakeTenantAuthSettingsRepository:
        async def get(self, session, *, tenant_id, default_issuer=None):
            return SimpleNamespace(issuer=default_issuer, suspended=True)

    monkeypatch.setattr(tenant_suspension_module, "control_session", _fake_control_session)
    monkeypatch.setattr(
        tenant_suspension_module,
        "TenantAuthSettingsRepository",
        _FakeTenantAuthSettingsRepository,
    )


async def test_run_assistant_rejects_a_suspended_tenant_before_any_tool_runs(monkeypatch, ctx):
    _suspend(monkeypatch)

    async def _boom(ctx, query, limit):
        pytest.fail("no tool must run for a suspended tenant")

    with pytest.raises(TenantSuspendedError):
        await run_assistant("hi", AssistantDeps(ctx=ctx, search=_boom))


async def test_stream_assistant_rejects_a_suspended_tenant_before_opening_the_stream(
    monkeypatch, ctx
):
    _suspend(monkeypatch)

    async def _boom(ctx, query, limit):
        pytest.fail("no tool must run for a suspended tenant")

    with pytest.raises(TenantSuspendedError):
        async with stream_assistant("hi", AssistantDeps(ctx=ctx, search=_boom)):
            pytest.fail("the stream must never open for a suspended tenant")
