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
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from pydantic_ai.ui.vercel_ai.request_types import FileUIPart, TextUIPart, UIMessage

from app.agents.assistant import AssistantDeps, chat_assistant, resolve_chat_model
from app.api.agents import ROUTING_ERRORS, routing_error_detail
from app.deps import Context
from app.request_limit import RequestLimit
from app.run_limits import RunDeadlineExceeded, RunLimits, build_run_limits, run_deadline

router = APIRouter(prefix="/api", tags=["chat"])

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


async def _bounded_by_deadline(source: AsyncIterator[str], limits: RunLimits) -> AsyncIterator[str]:
    """Wrap the adapter's already-encoded response stream in the run's wall-clock deadline.

    `dispatch_request()` builds and starts the run internally, so `run_deadline()` can no longer
    wrap the call that started it (unlike the one-shot and `/assistant/stream` endpoints); this
    is the next-outermost point still under our control. On timeout, emits one Vercel AI SDK
    `error` chunk — the same shape `pydantic_ai`'s own adapter emits for an in-run exception like
    `UsageLimitExceeded` — so the client sees one mapped error, never a raw exception or a
    connection that just stops.
    """
    try:
        async with run_deadline(limits):
            async for chunk in source:
                yield chunk
    except RunDeadlineExceeded as exc:
        payload = json.dumps({"type": "error", "errorText": str(exc)}, separators=(",", ":"))
        yield f"data: {payload}\n\n"


@router.post("/chat")
async def chat(request: Request, ctx: Context, _limit: RequestLimit) -> Response:
    if int(request.headers.get("content-length") or 0) > MAX_BODY_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Request body too large")
    deps = AssistantDeps(ctx=ctx)
    limits = build_run_limits()

    # Resolved before the adapter even parses the body (Spec 8 / #61, ADR-0008): a tenant with no
    # usable residency, an unlisted model, or no gateway credential is refused cleanly, never
    # silently served from `app.llm.get_model()`'s deployment-wide default.
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
    history = await deps.load_history(ctx, adapter.conversation_id or "")

    response = adapter.streaming_response(
        adapter.run_stream(
            message_history=history,
            deps=deps,
            model=model,
            usage_limits=limits.usage_limits,
            metadata=ctx.trace_attributes(),
        )
    )
    if isinstance(response, StreamingResponse):
        response.body_iterator = _bounded_by_deadline(response.body_iterator, limits)
    return response
