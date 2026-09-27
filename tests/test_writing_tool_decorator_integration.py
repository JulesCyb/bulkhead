"""ASGI-seam tests for the `writing_tool` decorator itself (spec A3 / #93, this ticket #109),
against PostgreSQL + pgvector (`pgserver`, via `uv sync --group dbtest`). Pattern:
`tests/test_writing_tool_approval_integration.py`, whose `_headers`/`_chat_path`/
`_audit_kinds_for_tenant` are reused directly here (that file's own module docstring documents
the pattern of importing its helpers rather than duplicating them).

`rename_document` is not the only thing exercised by `app/agents/writing_tools.py`'s decorator --
these tests register a *second*, throwaway writing tool on the real `chat_assistant` (exactly what
a developer following CLAUDE.md rule 4 would do for their own first writing tool: apply
`@writing_tool(chat_assistant)` and nothing else) for the duration of one test, drive it through
`POST /v1/t/{tenant_id}/api/chat`, and remove it again in teardown so no other test ever sees it.
This proves the decorator -- not `rename_document`'s own body -- is what produces the
`requested`/`approved`/`executed` audit trail for a normal write, and `failed_to_execute` for a
body that raises, generalizing beyond the one worked example.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.toolsets.function import FunctionToolset

from app.agents.assistant import AssistantDeps, chat_assistant
from app.agents.writing_tools import writing_tool
from app.main import app

pgserver = pytest.importorskip("pgserver")

from tests.support import cluster, environment, seed_conversation, seed_tenant  # noqa: E402
from tests.test_writing_tool_approval_integration import (  # noqa: E402
    _audit_kinds_for_tenant,
    _chat_path,
    _headers,
    _pending_action_rows,
)

_ = (cluster, environment)

CONVERSATION_ID = "conv-throwaway-1"
TOOL_NAME = "throwaway_write"
TOOL_CALL_ID = "call-throwaway-1"


def _function_toolset(agent) -> FunctionToolset:
    for toolset in agent.toolsets:
        if isinstance(toolset, FunctionToolset):
            return toolset
    raise AssertionError(f"{agent} has no function toolset")


@pytest.fixture
def throwaway_writing_tool():
    """Registers `throwaway_write` on the real `chat_assistant` via `@writing_tool(chat_assistant)`
    -- the same call a second real writing tool would make -- for the duration of one test, and
    removes it again on teardown (even on failure) so no other test in the suite ever sees it.

    The body itself does no real write: it reports whatever outcome the test's own `outcome`
    argument asks for, so one throwaway tool covers every audit-trail shape the decorator can
    produce (`executed`, `failed_to_execute` via `None`, `failed_to_execute` via a raise).
    """

    @writing_tool(chat_assistant)
    async def throwaway_write(ctx: RunContext[AssistantDeps], outcome: str) -> str | None:
        """A throwaway writing tool, for this test file only. Requires an approval from the
        asking member before it runs (ADR-0007), exactly like every other writing tool.

        Args:
            outcome: one of "succeed", "not_found", "fail" -- what the test wants this call to do.
        """
        if outcome == "fail":
            raise RuntimeError("the throwaway tool always fails on purpose")
        if outcome == "not_found":
            return None
        return "the throwaway write happened"

    try:
        yield TOOL_NAME
    finally:
        _function_toolset(chat_assistant).tools.pop(TOOL_NAME, None)


def _resolved(messages, tool_call_id: str) -> bool:
    for message in messages:
        for part in getattr(message, "parts", []):
            if (
                getattr(part, "tool_call_id", None) == tool_call_id
                and getattr(part, "part_kind", None) == "tool-return"
            ):
                return True
    return False


def _throwaway_model(*, outcome: str, tool_call_id: str = TOOL_CALL_ID) -> FunctionModel:
    """Calls `throwaway_write(outcome)` once, with a fixed `tool_call_id`, then -- once that
    call's `ToolReturnPart` shows up in history -- answers with plain text. Pattern:
    `tests.test_writing_tool_approval_integration._rename_model`."""
    args = {"outcome": outcome}

    async def call(messages, info: AgentInfo) -> ModelResponse:
        if _resolved(messages, tool_call_id):
            return ModelResponse(parts=[TextPart("Done.")])
        return ModelResponse(
            parts=[ToolCallPart(tool_name=TOOL_NAME, args=args, tool_call_id=tool_call_id)]
        )

    async def stream_call(messages, info: AgentInfo):
        if _resolved(messages, tool_call_id):
            yield "Done."
        else:
            yield {
                0: DeltaToolCall(
                    name=TOOL_NAME, json_args=json.dumps(args), tool_call_id=tool_call_id
                )
            }

    return FunctionModel(call, stream_function=stream_call)


def _propose_body(outcome: str) -> dict:
    return {
        "id": CONVERSATION_ID,
        "trigger": "submit-message",
        "messages": [
            {
                "id": "m1",
                "role": "user",
                "parts": [{"type": "text", "text": f"please {outcome}"}],
            }
        ],
    }


def _resume_body(*, tool_call_id: str, outcome: str, approved: bool) -> dict:
    return {
        "id": CONVERSATION_ID,
        "trigger": "submit-message",
        "messages": [
            {
                "id": "m2",
                "role": "assistant",
                "parts": [
                    {
                        "type": f"tool-{TOOL_NAME}",
                        "toolCallId": tool_call_id,
                        "state": "approval-responded",
                        "input": {"outcome": outcome},
                        "approval": {"id": f"approval-{tool_call_id}", "approved": approved},
                    }
                ],
            }
        ],
    }


@pytest.fixture
def client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_throwaway_writing_tool_executed_produces_the_same_audit_trail_as_rename_document(
    environment, client, use_model, throwaway_writing_tool
):
    """AC2 of #109: a second writing tool, registered with nothing but the decorator, produces
    the same `requested`/`approved`/`executed` audit trail `rename_document` does -- proving the
    trail comes from the decorator, not from anything specific to that one tool's own body."""
    tenant = await seed_tenant(environment, roles=["member"], via_operator=False)
    identity_id = tenant.identities["member"]
    await seed_conversation(
        environment,
        tenant_id=tenant.tenant_id,
        identity_id=identity_id,
        conversation_id=CONVERSATION_ID,
    )

    use_model(_throwaway_model(outcome="succeed"))

    async with client:
        proposed = await client.post(
            _chat_path(tenant.tenant_id),
            json=_propose_body("succeed"),
            headers=_headers(identity_id),
        )
        assert proposed.status_code == 200, proposed.text
        assert '"type":"tool-approval-request"' in proposed.text

        resumed = await client.post(
            _chat_path(tenant.tenant_id),
            json=_resume_body(tool_call_id=TOOL_CALL_ID, outcome="succeed", approved=True),
            headers=_headers(identity_id),
        )
        assert resumed.status_code == 200, resumed.text

    assert "the throwaway write happened" in resumed.text
    kinds = await _audit_kinds_for_tenant(environment.superuser_url, tenant_id=tenant.tenant_id)
    assert kinds == ["requested", "approved", "executed"]


async def test_throwaway_writing_tool_that_raises_is_recorded_as_failed_to_execute(
    environment, client, use_model, throwaway_writing_tool
):
    """A body that raises is recorded as `failed_to_execute` (the decorator's `except` branch)
    and the exception surfaces as the run's own mapped error chunk, exactly like any other
    in-run failure (`app/agents/run.py`) -- never a silently swallowed write."""
    tenant = await seed_tenant(environment, roles=["member"], via_operator=False)
    identity_id = tenant.identities["member"]
    await seed_conversation(
        environment,
        tenant_id=tenant.tenant_id,
        identity_id=identity_id,
        conversation_id=CONVERSATION_ID,
    )

    use_model(_throwaway_model(outcome="fail"))

    async with client:
        proposed = await client.post(
            _chat_path(tenant.tenant_id), json=_propose_body("fail"), headers=_headers(identity_id)
        )
        assert proposed.status_code == 200, proposed.text

        resumed = await client.post(
            _chat_path(tenant.tenant_id),
            json=_resume_body(tool_call_id=TOOL_CALL_ID, outcome="fail", approved=True),
            headers=_headers(identity_id),
        )

    assert resumed.status_code == 200, resumed.text
    assert '"type":"error"' in resumed.text

    kinds = await _audit_kinds_for_tenant(environment.superuser_url, tenant_id=tenant.tenant_id)
    assert kinds == ["requested", "approved", "failed_to_execute"]
    # #82: the pending action records the failed execution too, not only the audit trail.
    rows = await _pending_action_rows(environment.superuser_url, tenant_id=tenant.tenant_id)
    assert [row["status"] for row in rows] == ["execution_failed"]


async def test_throwaway_writing_tool_nothing_found_is_also_failed_to_execute(
    environment, client, use_model, throwaway_writing_tool
):
    """`None` (the body's own "nothing to act on" sentinel) is recorded as `failed_to_execute`
    without raising, and the model still gets a normal (non-error) reply -- the decorator's other
    non-exception branch, distinct from a raised exception."""
    tenant = await seed_tenant(environment, roles=["member"], via_operator=False)
    identity_id = tenant.identities["member"]
    await seed_conversation(
        environment,
        tenant_id=tenant.tenant_id,
        identity_id=identity_id,
        conversation_id=CONVERSATION_ID,
    )

    use_model(_throwaway_model(outcome="not_found"))

    async with client:
        proposed = await client.post(
            _chat_path(tenant.tenant_id),
            json=_propose_body("not_found"),
            headers=_headers(identity_id),
        )
        assert proposed.status_code == 200, proposed.text

        resumed = await client.post(
            _chat_path(tenant.tenant_id),
            json=_resume_body(tool_call_id=TOOL_CALL_ID, outcome="not_found", approved=True),
            headers=_headers(identity_id),
        )

    assert resumed.status_code == 200, resumed.text
    assert '"type":"error"' not in resumed.text

    kinds = await _audit_kinds_for_tenant(environment.superuser_url, tenant_id=tenant.tenant_id)
    assert kinds == ["requested", "approved", "failed_to_execute"]
