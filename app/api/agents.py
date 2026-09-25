"""Agent endpoints: one-shot answer and text stream (SSE).

For a Vercel AI SDK chat UI, see app/api/chat.py.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_ai.exceptions import UsageLimitExceeded

from app.agents.assistant import AssistantDeps, run_assistant, stream_assistant
from app.deps import Context
from app.run_limits import RunDeadlineExceeded, build_run_limits, run_deadline


def _sse(data: str) -> str:
    """Frame text as one SSE event. Multi-line deltas become multiple data: lines,
    which the client reassembles with newlines — raw interpolation would silently
    drop every line lacking the data: prefix.
    """
    return "".join(f"data: {line}\n" for line in data.split("\n")) + "\n"


def _sse_error(error: str, message: str) -> str:
    """Frame a terminal `event: error` SSE event — the mapped, clean end to a run that hit its
    run limit, distinct from a silent stop or a raw exception. `message` is deliberately generic
    (see `RunDeadlineExceeded`); it never carries the underlying provider's own error text.
    """
    payload = json.dumps({"error": error, "message": message})
    return f"event: error\ndata: {payload}\n\n"


router = APIRouter(prefix="/agents", tags=["agents"])


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)


class RunResponse(BaseModel):
    output: str


@router.post("/assistant/run", response_model=RunResponse)
async def run(body: RunRequest, ctx: Context) -> RunResponse:
    try:
        output = await run_assistant(body.prompt, AssistantDeps(ctx=ctx))
    except UsageLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail={"error": "run_limit_exceeded", "message": str(exc)},
        ) from exc
    except RunDeadlineExceeded as exc:
        raise HTTPException(
            status_code=504,
            detail={"error": "run_deadline_exceeded", "message": str(exc)},
        ) from exc
    return RunResponse(output=output)


@router.post("/assistant/stream")
async def stream(body: RunRequest, ctx: Context) -> StreamingResponse:
    limits = build_run_limits()

    async def events() -> AsyncIterator[str]:
        try:
            # The deadline must bound opening the stream AND reading every delta from it, not
            # just the call that starts it — see run_deadline()'s docstring.
            async with run_deadline(limits):
                async with stream_assistant(body.prompt, AssistantDeps(ctx=ctx), limits) as result:
                    async for delta in result.stream_text(delta=True):
                        yield _sse(delta)
        except UsageLimitExceeded as exc:
            yield _sse_error("run_limit_exceeded", str(exc))
            return
        except RunDeadlineExceeded as exc:
            yield _sse_error("run_deadline_exceeded", str(exc))
            return
        yield "event: done\ndata: \n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
