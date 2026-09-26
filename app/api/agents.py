"""Agent endpoints: one-shot answer and text stream (SSE).

Transport only (spec A3 / #93, #107): each route parses the prompt, prepares the run
(`app.agents.run.prepare_run` -- model, run limit and deadline, tracing, tool dependencies, all
from the request's context and its tenant record), calls one execution method, and renders the
result or the mapped error (`app.agents.run_errors`). Both methods bind the reading-only agent
(ADR-0007); nothing here wraps a deadline or span attributes around anything.

For a Vercel AI SDK chat UI, see app/api/chat.py.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agents.run import prepare_run
from app.agents.run_errors import (
    MAPPED_RUN_ERRORS,
    run_error_http_exception,
    run_error_sse_event,
    sse,
)
from app.deps import Context
from app.request_limit import RequestLimit

router = APIRouter(prefix="/agents", tags=["agents"])


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)


class RunResponse(BaseModel):
    output: str


@router.post("/assistant/run", response_model=RunResponse)
async def run(body: RunRequest, ctx: Context, _limit: RequestLimit) -> RunResponse:
    try:
        prepared = await prepare_run(ctx)
        output = await prepared.answer(body.prompt)
    except MAPPED_RUN_ERRORS as exc:
        raise run_error_http_exception(exc) from exc
    return RunResponse(output=output)


@router.post("/assistant/stream")
async def stream(body: RunRequest, ctx: Context, _limit: RequestLimit) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        # Preparation happens in here, not before the response: by the time any failure --
        # a routing failure included -- is known, the response has started (status 200), so it
        # ends the stream as a mapped `event: error`, never a raw exception.
        try:
            prepared = await prepare_run(ctx)
            async for delta in prepared.stream_text(body.prompt):
                yield sse(delta)
        except MAPPED_RUN_ERRORS as exc:
            yield run_error_sse_event(exc)
            return
        yield "event: done\ndata: \n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
