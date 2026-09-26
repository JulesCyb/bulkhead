"""Unit tests proving the agent-run entry points (`run_assistant`/`stream_assistant`,
`app/agents/assistant.py`) carry no suspension check of their own (#106, ADR-0010): suspension has
exactly two enforcement points project-wide (`app/db/session.py`'s module docstring) -- context
resolution, for a request whose `RequestContext` therefore never reaches this module suspended in
the first place, and `tenant_session()`'s own routing read, for a record-less context (a job, a
test, or any caller context resolution never built). This file exercises the second one directly:
a record-less `ctx` (the shared `ctx` fixture) whose tool opens a real `tenant_session()`, with
`app.db.session._resolve_tenant_alias` made to behave exactly as it would against a suspended
tenant's own control-plane row -- no real database, no ASGI request, unlike `tests/test_api.py`'s
HTTP-level coverage of the same two endpoints and `tests/test_tenant_suspension_asgi.py`'s coverage
of the first enforcement point.
"""

from __future__ import annotations

import pytest

import app.db.session as session_module
from app.agents.assistant import AssistantDeps, run_assistant, stream_assistant
from app.db.session import TenantSuspendedError, tenant_session


def _fail_the_session_routing_read(monkeypatch: pytest.MonkeyPatch, tenant_id) -> None:
    """Stands in for `_resolve_tenant_alias`'s real behaviour against a suspended tenant's own
    `control.tenants_view` row -- raising before `tenant_session()` ever opens a session against
    the tenant's actual data, exactly like the real routing read (`app/db/session.py`)."""

    async def _raise(ctx):
        assert ctx.tenant_id == tenant_id
        raise TenantSuspendedError(ctx.tenant_id)

    monkeypatch.setattr(session_module, "_resolve_tenant_alias", _raise)


async def _search_opens_a_session(ctx, query, limit):
    """A stand-in for a real tool's repository call: opens `tenant_session(ctx)` itself, exactly
    as `app.tools.documents.search_documents` does, rather than returning a canned result."""
    async with tenant_session(ctx):
        pytest.fail("the session must never actually open for a suspended tenant")
    return []  # pragma: no cover -- unreachable; tenant_session raises before yielding


async def test_run_assistant_has_no_suspension_check_of_its_own_and_the_session_layer_raises(
    ctx, monkeypatch, test_model
):
    """`run_assistant` no longer checks suspension itself (#106): a `TenantSuspendedError` raised
    deep inside a tool's own `tenant_session()` call -- the record-less context's one enforcement
    point -- propagates out of `run_assistant` untouched, not caught or duplicated by a check of
    its own. `test_model` (`tests/conftest.py`) drives the one-shot agent through exactly one call
    to `search_documents`, and patches `resolve_chat_model` so model resolution never depends on
    the record-less `ctx` this test is about."""
    _fail_the_session_routing_read(monkeypatch, ctx.tenant_id)

    with pytest.raises(TenantSuspendedError) as exc_info:
        await run_assistant("hi", AssistantDeps(ctx=ctx, search=_search_opens_a_session))
    assert exc_info.value.tenant_id == ctx.tenant_id


async def test_stream_assistant_has_no_suspension_check_of_its_own_and_the_session_layer_raises(
    ctx, monkeypatch, test_model
):
    """Same proof for the streaming entry point."""
    _fail_the_session_routing_read(monkeypatch, ctx.tenant_id)

    with pytest.raises(TenantSuspendedError) as exc_info:
        async with stream_assistant("hi", AssistantDeps(ctx=ctx, search=_search_opens_a_session)):
            pytest.fail("the stream must never open once a tool call raises")
    assert exc_info.value.tenant_id == ctx.tenant_id


async def test_run_assistant_reaches_the_tool_when_nothing_is_suspended(ctx, test_model, calls):
    """Negative control: with no suspension anywhere, `run_assistant` actually reaches the
    injected `search` -- proving the failure the two tests above rely on comes from inside the
    tool call, not from some other, earlier check this rewrite forgot to remove."""

    async def _search_records_it_ran(ctx, query, limit):
        calls.append((ctx.tenant_id, query, limit))
        return []

    await run_assistant("hi", AssistantDeps(ctx=ctx, search=_search_records_it_ran))
    assert calls and calls[0][0] == ctx.tenant_id
