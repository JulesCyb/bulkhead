"""Chat endpoint in the Vercel AI SDK format.

A Next.js frontend using `useChat` (Vercel AI SDK) can talk directly to
POST /v1/t/{tenant_id}/api/chat — PydanticAI translates messages and stream (incl. tool
events). See docs/frontend.md.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

from app.agents.assistant import AssistantDeps, chat_assistant
from app.db.session import TenantSuspendedError
from app.deps import FORBIDDEN_DETAIL, Context
from app.llm import get_model
from app.request_limit import RequestLimit
from app.run_limits import RunDeadlineExceeded, RunLimits, build_run_limits, run_deadline
from app.tenant_suspension import ensure_tenant_not_suspended

router = APIRouter(prefix="/api", tags=["chat"])

# Chat histories are legitimately larger than a single prompt, but not unbounded —
# mirrors the 20k prompt cap on /v1/t/{tenant_id}/agents/assistant/*.
MAX_BODY_BYTES = 200_000


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
    # Independent of deps.py's own check (Spec 9 / #69, ADR-0010): this is "the agent-run entry
    # point", checked on its own before the chat-capable (writing-tool) agent ever dispatches.
    try:
        await ensure_tenant_not_suspended(ctx)
    except TenantSuspendedError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, FORBIDDEN_DETAIL) from exc
    deps = AssistantDeps(ctx=ctx)
    limits = build_run_limits()
    response = await VercelAIAdapter.dispatch_request(
        request,
        agent=chat_assistant,
        deps=deps,
        model=get_model(deps.model_name),
        usage_limits=limits.usage_limits,
        metadata=ctx.trace_attributes(),
        sdk_version=6,
    )
    if isinstance(response, StreamingResponse):
        response.body_iterator = _bounded_by_deadline(response.body_iterator, limits)
    return response
