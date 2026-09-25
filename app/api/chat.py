"""Chat endpoint in the Vercel AI SDK format.

A Next.js frontend using `useChat` (Vercel AI SDK) can talk directly to
POST /v1/t/{tenant_id}/api/chat — PydanticAI translates messages and stream (incl. tool
events). See docs/frontend.md.

ADR-0006 / Spec 4 (#33): the only client input this endpoint trusts is the newest message's
member-authored content. `VercelAIAdapter.dispatch_request()` cannot be used as-is for that —
its own `messages` property (and therefore the history it feeds the agent) is built from
*every* message in the request body, only stripped of system prompts and disallowed file
references (`UIAdapter.sanitize_messages`), never of earlier turns or of a forged
assistant/tool part riding along on the newest one. So this endpoint builds the adapter itself,
loads the real history from `ConversationsRepository` (via `AssistantDeps.load_history`), and
overrides the adapter's `messages` with just the newest, member-authored turn before running —
see `_member_authored_messages()` below.

ADR-0006 / Spec 4 (#34): once the run completes, its newly produced messages are written back
through `AssistantDeps.save_run` (the real `ConversationsRepository`, in a session of its own).
Persistence is wired through the adapter's own `on_complete` hook, which fires when the run
itself finishes — not when the client has read the last byte of the SSE response — but the
`StreamingResponse.body_iterator` a caller stops reading from is a generator that simply never
advances past the point the caller stopped, so persistence would still depend on the client
finishing the read unless something keeps driving the underlying stream. `_decouple_from_client`
below does that: it drains the adapter's stream into a queue from a background task the ASGI
server's own consumption can't stall, so persistence and the client's read of the stream are two
independent things, exactly as ADR-0006 requires. The conversation id reaching the run itself is
combined with the tenant id (`_tenant_scoped_conversation_id`), because tracing has no
Row-Level Security to fall back on if two tenants' clients ever pick the same id.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Coroutine
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import ModelMessage
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from pydantic_ai.ui.vercel_ai.request_types import FileUIPart, TextUIPart, UIMessage

from app.agents.assistant import AssistantDeps, chat_assistant, resolve_chat_model
from app.api.agents import ROUTING_ERRORS, routing_error_detail
from app.context import RequestContext
from app.db.session import TenantSuspendedError
from app.deps import FORBIDDEN_DETAIL, Context
from app.observability import (
    instrumentation_capabilities,
    resolve_tenant_tracing,
    tenant_span_attributes,
)
from app.request_limit import RequestLimit
from app.run_limits import RunDeadlineExceeded, RunLimits, build_run_limits, run_deadline
from app.tenant_suspension import ensure_tenant_not_suspended
from app.tools.approvals import resolve_incoming_decisions

router = APIRouter(prefix="/api", tags=["chat"])

# Fire-and-forget background tasks (the stream-draining task started by `_decouple_from_client`)
# are only weakly referenced by the event loop once nothing else holds them — keeping a strong
# reference here until each finishes is what stops asyncio from garbage-collecting one mid-flight.
_background_tasks: set[asyncio.Task[None]] = set()


def _spawn_background(coro: Coroutine[Any, Any, None]) -> None:
    task = asyncio.ensure_future(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _drain_into_queue(source: AsyncIterator[str], queue: asyncio.Queue[str | None]) -> None:
    """Pulls every chunk out of `source` — driving it to exhaustion, including whatever
    `on_complete` callback fires on its last item — regardless of whether anything is still
    reading the other end of `queue`. Errors already surface as an encoded `error` chunk
    upstream (`_bounded_by_deadline`, the adapter's own `on_error`), so nothing here is expected
    to raise; the `except` is a last-resort guard against an unretrieved-task-exception warning
    if one somehow does, not a place that swallows a persistence failure.
    """
    try:
        async for chunk in source:
            await queue.put(chunk)
    except Exception:
        pass
    finally:
        await queue.put(None)


async def _queue_iterator(queue: asyncio.Queue[str | None]) -> AsyncIterator[str]:
    while (item := await queue.get()) is not None:
        yield item


def _decouple_from_client(source: AsyncIterator[str]) -> AsyncIterator[str]:
    """Returns an iterator fed from a background task that drains `source` on its own, so a
    client that stops reading the response never stalls `source` itself (ADR-0006, #34) — in
    particular, the `on_complete` persistence hook attached to `source` still runs to completion.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    _spawn_background(_drain_into_queue(source, queue))
    return _queue_iterator(queue)


# Chat histories are legitimately larger than a single prompt, but not unbounded —
# mirrors the 20k prompt cap on /v1/t/{tenant_id}/agents/assistant/*.
MAX_BODY_BYTES = 200_000


def _member_authored_messages(messages: list[UIMessage]) -> list[ModelMessage]:
    """The only part of the request body this endpoint trusts as new input.

    Takes the newest message and, only if it is the member's own turn (`role == "user"`),
    keeps just the part types a member can actually author — text and file parts — before
    handing it to pydantic-ai's own message loader. Everything else is discarded before it is
    ever parsed into a `ModelMessage`: every earlier turn of any role (the true history comes
    from `ConversationsRepository` instead), and any assistant or tool part a forged request
    spliced onto the newest turn itself (e.g. a fabricated `tool-search_documents` result made
    to look like something the assistant already said).
    """
    if not messages:
        return []
    newest = messages[-1]
    if newest.role != "user":
        return []
    member_parts = [p for p in newest.parts if isinstance(p, (TextUIPart, FileUIPart))]
    if not member_parts:
        return []
    filtered = newest.model_copy(update={"parts": member_parts})
    return VercelAIAdapter.load_messages([filtered])


async def _bounded_by_deadline(
    source: AsyncIterator[str], limits: RunLimits, ctx: RequestContext
) -> AsyncIterator[str]:
    """Wrap the adapter's already-encoded response stream in the run's wall-clock deadline, and
    (Spec 8 / #62) in this tenant's span attributes for as long as the stream is actually
    consumed — neither `run_deadline()` nor `tenant_span_attributes()` can wrap the call that
    starts the run (`adapter.run_stream()` builds an async generator that only does its work once
    iterated, and this endpoint no longer calls `dispatch_request()` directly — see module
    docstring); this is the next-outermost point still under our control, and the only one that
    covers the plain async generator's full resumed lifetime. On timeout, emits one Vercel AI SDK
    `error` chunk — the same shape `pydantic_ai`'s own adapter emits for an in-run exception like
    `UsageLimitExceeded` — so the client sees one mapped error, never a raw exception or a
    connection that just stops.
    """
    try:
        async with run_deadline(limits):
            with tenant_span_attributes(ctx.trace_attributes()):
                async for chunk in source:
                    yield chunk
    except RunDeadlineExceeded as exc:
        payload = json.dumps({"type": "error", "errorText": str(exc)}, separators=(",", ":"))
        yield f"data: {payload}\n\n"


@router.post("/chat")
async def chat(request: Request, ctx: Context, _limit: RequestLimit) -> Response:
    if int(request.headers.get("content-length") or 0) > MAX_BODY_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Request body too large")
    # Independent of deps.py's own check (Spec 9 / #69, ADR-0010): this is "the agent-run entry
    # point", checked on its own before the chat-capable (writing-tool) agent ever dispatches.
    try:
        await ensure_tenant_not_suspended(ctx.tenant_id)
    except TenantSuspendedError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, FORBIDDEN_DETAIL) from exc
    tracing = await resolve_tenant_tracing(ctx)
    deps = AssistantDeps(
        ctx=ctx,
        residency=tracing.residency,
        content_tracing_opt_in=tracing.content_tracing_opt_in,
    )
    limits = build_run_limits()

    # Resolved before the adapter even parses the body (Spec 8 / #61, ADR-0008): a tenant with no
    # usable residency, an unlisted model, or no gateway credential is refused cleanly, never
    # silently served from the removed, deployment-wide `app.llm.get_model()` (ai-app-starter#7).
    try:
        model = await resolve_chat_model(deps)
    except ROUTING_ERRORS as exc:
        raise HTTPException(status_code=503, detail=routing_error_detail(exc)) from exc

    try:
        adapter = await VercelAIAdapter.from_request(request, agent=chat_assistant, sdk_version=6)
    except ValidationError as exc:
        # Mirrors VercelAIAdapter.dispatch_request()'s own handling of a malformed body — this
        # endpoint no longer calls dispatch_request() directly (see module docstring), so that
        # behavior has to be reproduced here rather than inherited.
        try:
            content = exc.json()
        except ValueError:
            content = exc.json(include_input=False)
        return Response(
            content=content,
            media_type="application/json",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )
    # Overrides the adapter's own `messages` cached property (normally every message in the
    # body) with just the newest, member-authored turn — see module docstring and
    # `_member_authored_messages()`.
    adapter.messages = _member_authored_messages(adapter.run_input.messages)

    assert deps.load_history is not None
    assert deps.save_run is not None
    conversation_id = adapter.conversation_id or ""
    deps.conversation_id = conversation_id
    history = await deps.load_history(ctx, conversation_id)

    # ADR-0007 / #40: a resumed request may carry the member's approve/refuse decision for a
    # writing tool's earlier deferred call (`VercelAIAdapter.deferred_tool_results`, extracted
    # from the raw request body regardless of the `adapter.messages` override above). A refusal
    # is resolved by pydantic-ai substituting `ToolDenied` directly -- the tool's own
    # `args_validator` never runs for a denied call -- so this is the only place a refusal's
    # audit milestone can ever be recorded; an approval is resolved here too, before the run
    # itself starts, so the tool's own execution-time re-verification always has an `approved`
    # (not merely `pending`) record to check against.
    deferred_tool_results = adapter.deferred_tool_results
    if deferred_tool_results is not None:
        await resolve_incoming_decisions(
            ctx, conversation_id=conversation_id, decisions=deferred_tool_results.approvals
        )

    async def _persist_new_messages(result: AgentRunResult[str]) -> None:
        # `on_complete` only fires when the run finishes successfully (see module docstring) —
        # a run that raises before completing never reaches here, so nothing is persisted for it.
        assert deps.save_run is not None
        await deps.save_run(ctx, conversation_id, result.new_messages())

    capabilities = instrumentation_capabilities(tracing.residency, tracing.content_tracing_opt_in)
    with tenant_span_attributes(ctx.trace_attributes()):
        response = adapter.streaming_response(
            adapter.run_stream(
                message_history=history,
                deps=deps,
                model=model,
                usage_limits=limits.usage_limits,
                metadata=ctx.trace_attributes(),
                capabilities=capabilities,
                # Tenant-scoped, not the client's bare id — two tenants whose clients happen to
                # pick the same conversation id must never look like the same trace subject
                # (ADR-0006).
                conversation_id=(f"{ctx.tenant_id}:{conversation_id}" if conversation_id else None),
                on_complete=_persist_new_messages,
            )
        )
    if isinstance(response, StreamingResponse):
        response.body_iterator = _decouple_from_client(
            _bounded_by_deadline(response.body_iterator, limits, ctx)
        )
    return response
