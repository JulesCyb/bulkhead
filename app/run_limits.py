"""The ceiling every agent run carries, by default, with no per-agent wiring required.

ADR-0009 / CONTEXT.md's "Run limit": the ceiling of model requests, tool calls, and wall-clock
time a single agent run may consume, enforced inside the application. It protects against loops
— for example one a poisoned document provokes — never against cost; a tenant's budget is
enforced outside the application, at the gateway (see the gateway-credential tickets).

`build_run_limits()` is the one place `pydantic_ai.usage.UsageLimits` gets constructed; call it
once per run and pass its `usage_limits` to the run, and wrap the call that starts the run — or
the loop that consumes its stream — in `run_deadline()`. Every entry point that starts a run does
both:
- `app/agents/run.py` (`prepare_run` builds the limit once; `PreparedRun.answer`/`stream_text`/
  `chat` wrap it) — the one-shot, streaming, and Vercel AI SDK chat endpoints alike; the deadline
  bounds the full open-and-consume lifecycle, not just the call that opens the stream, and no
  route wraps it.

Exceeding the request/tool-call/token ceiling raises `pydantic_ai.exceptions.UsageLimitExceeded`;
exceeding the wall-clock deadline raises `RunDeadlineExceeded` from this module. Both are plain
exceptions raised *during* the run, so the streaming entry points' own exception handling (ours in
`app/api/agents.py` and `app/agents/run.py`'s chat stream, PydanticAI's in the Vercel adapter for
the ceiling case) turns them into one clean, terminal error instead of a raw exception or a
silent stop.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from pydantic_ai.usage import UsageLimits

from app.config import Settings, get_settings


class RunDeadlineExceeded(Exception):
    """A single agent run exceeded its configured wall-clock deadline.

    Deliberately generic: it never carries the underlying provider's own error text, so a
    stalled or misbehaving provider's message never reaches a client — see ADR-0009's run-limit
    decision and the review finding "No wall-clock deadline on model calls."
    """

    def __init__(self) -> None:
        super().__init__("Agent run exceeded its wall-clock deadline.")


@dataclass(frozen=True)
class RunLimits:
    """The ceiling one agent run may consume — model requests, tool calls, tokens, and
    wall-clock time — built once from configuration by `build_run_limits()`."""

    usage_limits: UsageLimits
    deadline_seconds: float


def build_run_limits(settings: Settings | None = None) -> RunLimits:
    """Build this run's ceilings from configuration.

    Call once per run and pass the result at every place an agent run starts — never construct
    `UsageLimits` (or a deadline) ad hoc at a call site, so a new agent picks up this protection
    from its first commit without anyone remembering to wire it in.
    """
    s = settings or get_settings()
    return RunLimits(
        usage_limits=UsageLimits(
            request_limit=s.run_request_limit,
            tool_calls_limit=s.run_tool_calls_limit,
            total_tokens_limit=s.run_total_tokens_limit,
        ),
        deadline_seconds=s.run_deadline_seconds,
    )


@asynccontextmanager
async def run_deadline(limits: RunLimits) -> AsyncGenerator[None]:
    """Bound the wrapped code to the run's wall-clock deadline.

    Wrap the call into pydantic_ai *from the outside* — `await assistant.run(...)`, or the full
    `async for` loop consuming a `run_stream()` result — never a single call inside a `Model`'s
    own `request`/`request_stream` (a `WrapperModel` gets invoked from inside PydanticAI's own
    structured concurrency, sometimes from a child task it spawns itself; a cancel scope opened
    there can be closed out of the order PydanticAI's own nested scopes expect, raising anyio's
    "cancel scope that isn't the current task's" error instead of a clean timeout).

    `asyncio.timeout()` cancels whichever task is executing the `with` block. Per
    `pydantic_ai.exceptions.RunCancelled`'s own docs, this is the supported way to bound a run's
    wall-clock time: "External cancellation of the task running the agent ... is
    infrastructure-level and keeps propagating as `asyncio.CancelledError` ... This also works
    with the `TimeoutError` raised by `asyncio.timeout()` or `asyncio.wait_for()`." The
    cancellation crosses PydanticAI's internal (correctly nested) cancel scopes cleanly because
    it originates outside all of them, rather than injected into the middle of them.
    """
    try:
        async with asyncio.timeout(limits.deadline_seconds):
            yield
    except TimeoutError as exc:
        raise RunDeadlineExceeded from exc


async def bounded_by_deadline[T](source: AsyncIterator[T], limits: RunLimits) -> AsyncIterator[T]:
    """Wrap an already-started async iterator (e.g. an already-encoded response byte/text stream)
    so consuming it further never holds a client open past the run's wall-clock deadline.

    Used where the run itself is already underway by the time our code gets to iterate its
    output — the Vercel AI chat adapter builds and starts the run internally, so `run_deadline()`
    can no longer wrap the call that started it; wrapping the response stream it hands back is
    the next-outermost point still under our control.
    """
    async with run_deadline(limits):
        async for item in source:
            yield item


__all__ = [
    "RunDeadlineExceeded",
    "RunLimits",
    "bounded_by_deadline",
    "build_run_limits",
    "run_deadline",
]
