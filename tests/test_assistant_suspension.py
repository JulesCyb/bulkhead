"""Unit tests proving a prepared run (`app.agents.run`: preparation, `answer`, `stream_text`)
carries no suspension check of its own (#106, ADR-0010): suspension has exactly two enforcement
points project-wide (`app/db/session.py`'s module docstring) -- context resolution, which refuses a
suspended tenant's record before a `RequestContext` for it is ever built, and `tenant_session()`
itself, which refuses a record that says suspended (or, for a record-less context, its own routing
read). This file exercises the second one directly: a context whose record says suspended, whose
tool opens a real `tenant_session()` -- no real database, no ASGI request, unlike
`tests/test_api.py`'s HTTP-level coverage of the same two endpoints and
`tests/test_tenant_suspension_asgi.py`'s coverage of the first enforcement point. A record-less
context never reaches a run at all: preparation refuses it (`tests/test_run.py`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from pydantic_ai.models.test import TestModel

from app.agents.run import prepare_run
from app.db.session import TenantSuspendedError, tenant_session
from tests.conftest import returning


async def _collect(stream: AsyncIterator[str]) -> str:
    return "".join([delta async for delta in stream])


async def _search_opens_a_session(ctx, query, limit):
    """A stand-in for a real tool's repository call: opens `tenant_session(ctx)` itself, exactly
    as `app.tools.documents.search_documents` does, rather than returning a canned result."""
    async with tenant_session(ctx):
        pytest.fail("the session must never actually open for a suspended tenant")
    return []  # pragma: no cover -- unreachable; tenant_session raises before yielding


@pytest.fixture
def suspended_ctx(run_ctx):
    record = dataclasses.replace(
        run_ctx.tenant_record, suspended_at=datetime(2026, 1, 1, tzinfo=UTC)
    )
    return dataclasses.replace(run_ctx, tenant_record=record)


@pytest.mark.parametrize("execute", ["answer", "stream_text"])
async def test_the_run_has_no_suspension_check_of_its_own_and_the_session_layer_raises(
    suspended_ctx, execute
):
    """Suspension has exactly two enforcement points (`app/db/session.py`'s module docstring):
    context resolution, and `tenant_session()` itself. Preparation and execution add none -- a
    record that says suspended is prepared without complaint, and the `TenantSuspendedError` a
    tool's own `tenant_session()` raises propagates out of the execution untouched."""
    prepared = await prepare_run(
        suspended_ctx,
        model_resolver=returning(TestModel(call_tools=["search_documents"])),
        search=_search_opens_a_session,
    )

    with pytest.raises(TenantSuspendedError) as exc_info:
        if execute == "answer":
            await prepared.answer("hi")
        else:
            await _collect(prepared.stream_text("hi"))
    assert exc_info.value.tenant_id == suspended_ctx.tenant_id


async def test_the_run_reaches_the_tool_when_nothing_is_suspended(run_ctx, calls):
    """Negative control for the test above: the failure there comes from inside the tool call,
    not from an earlier check."""

    async def _search_records_it_ran(ctx, query, limit):
        calls.append((ctx.tenant_id, query, limit))
        return []

    prepared = await prepare_run(
        run_ctx,
        model_resolver=returning(TestModel(call_tools=["search_documents"])),
        search=_search_records_it_ran,
    )
    await prepared.answer("hi")
    assert calls and calls[0][0] == run_ctx.tenant_id
