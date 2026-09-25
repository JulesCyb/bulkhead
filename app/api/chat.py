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
from app.context import RequestContext
from app.deps import Context
from app.llm import get_model
from app.observability import (
    instrumentation_capabilities,
    resolve_tenant_tracing,
    tenant_span_attributes,
)
from app.request_limit import RequestLimit
from app.run_limits import RunDeadlineExceeded, RunLimits, build_run_limits, run_deadline

router = APIRouter(prefix="/api", tags=["chat"])

# Chat histories are legitimately larger than a single prompt, but not unbounded —
# mirrors the 20k prompt cap on /v1/t/{tenant_id}/agents/assistant/*.
MAX_BODY_BYTES = 200_000


async def _bounded_by_deadline(
    source: AsyncIterator[str], limits: RunLimits, ctx: RequestContext
) -> AsyncIterator[str]:
    """Wrap the adapter's already-encoded response stream in the run's wall-clock deadline, and
    (Spec 8 / #62) in this tenant's span attributes for as long as the stream is actually
    consumed — `dispatch_request()` builds and starts the run internally, so neither
    `run_deadline()` nor `tenant_span_attributes()` can wrap the call that started it (unlike the
    one-shot and `/assistant/stream` endpoints); this is the next-outermost point still under our
    control, and the only one that covers the plain async generator's full resumed lifetime. On
    timeout, emits one Vercel AI SDK `error` chunk — the same shape `pydantic_ai`'s own adapter
    emits for an in-run exception like `UsageLimitExceeded` — so the client sees one mapped
    error, never a raw exception or a connection that just stops.
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
    tracing = await resolve_tenant_tracing(ctx)
    deps = AssistantDeps(
        ctx=ctx,
        residency=tracing.residency,
        content_tracing_opt_in=tracing.content_tracing_opt_in,
    )
    limits = build_run_limits()
    capabilities = instrumentation_capabilities(tracing.residency, tracing.content_tracing_opt_in)
    with tenant_span_attributes(ctx.trace_attributes()):
        response = await VercelAIAdapter.dispatch_request(
            request,
            agent=chat_assistant,
            deps=deps,
            model=get_model(deps.model_name),
            usage_limits=limits.usage_limits,
            metadata=ctx.trace_attributes(),
            capabilities=capabilities,
            sdk_version=6,
        )
    if isinstance(response, StreamingResponse):
        response.body_iterator = _bounded_by_deadline(response.body_iterator, limits, ctx)
    return response
