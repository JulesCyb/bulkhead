"""Agent endpoints: one-shot answer and text stream (SSE).

For a Vercel AI SDK chat UI, see app/api/chat.py.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agents.assistant import AssistantDeps, run_assistant, stream_assistant
from app.deps import Context
from app.request_limit import RequestLimit


def _sse(data: str) -> str:
    """Frame text as one SSE event. Multi-line deltas become multiple data: lines,
    which the client reassembles with newlines — raw interpolation would silently
    drop every line lacking the data: prefix.
    """
    return "".join(f"data: {line}\n" for line in data.split("\n")) + "\n"


router = APIRouter(prefix="/agents", tags=["agents"])


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)


class RunResponse(BaseModel):
    output: str


@router.post("/assistant/run", response_model=RunResponse)
async def run(body: RunRequest, ctx: Context, _limit: RequestLimit) -> RunResponse:
    output = await run_assistant(body.prompt, AssistantDeps(ctx=ctx))
    return RunResponse(output=output)


@router.post("/assistant/stream")
async def stream(body: RunRequest, ctx: Context, _limit: RequestLimit) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        async with stream_assistant(body.prompt, AssistantDeps(ctx=ctx)) as result:
            async for delta in result.stream_text(delta=True):
                yield _sse(delta)
        yield "event: done\ndata: \n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
