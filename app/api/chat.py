"""Chat endpoint in the Vercel AI SDK format.

A Next.js frontend using `useChat` (Vercel AI SDK) can talk directly to POST /api/chat —
PydanticAI translates messages and stream (incl. tool events). See docs/frontend.md.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

from app.agents.assistant import AssistantDeps, assistant
from app.deps import Context
from app.llm import get_model
from app.request_limit import RequestLimit

router = APIRouter(prefix="/api", tags=["chat"])

# Chat histories are legitimately larger than a single prompt, but not unbounded —
# mirrors the 20k prompt cap on /agents/assistant/*.
MAX_BODY_BYTES = 200_000


@router.post("/chat")
async def chat(request: Request, ctx: Context, _limit: RequestLimit) -> Response:
    if int(request.headers.get("content-length") or 0) > MAX_BODY_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Request body too large")
    deps = AssistantDeps(ctx=ctx)
    return await VercelAIAdapter.dispatch_request(
        request,
        agent=assistant,
        deps=deps,
        model=get_model(deps.model_name),
        metadata=ctx.trace_attributes(),
        sdk_version=6,
    )
