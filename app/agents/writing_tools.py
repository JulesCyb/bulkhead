"""One decorator for a writing tool (spec A3 / #93, this ticket #109): registering the
`args_validator=require_approval` hook (ADR-0007, `app/tools/approvals.py`) and wrapping the
body's read/clear/execute/record sequence used to be two things a new writing tool had to copy by
hand (CLAUDE.md rule 4's old "copy this shape verbatim" instruction). Both are now one call:

    @writing_tool(chat_assistant)
    async def some_tool(ctx: RunContext[AssistantDeps], **args) -> str | None:
        ...  # the one write, through the repository layer, nothing else
        return "text the model sees" if it_worked else None  # None = "nothing to act on"

`rename_document` (`app/agents/assistant.py`) is the worked example: its body does the one
repository call and returns either the success text or `None`, and nothing else -- no read of
`ctx.deps.pending_approval`, no `try`/`except`, no call to `record_write_outcome`.

What `writing_tool(agent)` does, once, when applied to a body function `fn`:

1. **Registers** the wrapped function on `agent` with `args_validator=require_approval` -- the
   two-pass hook (`app/tools/approvals.py`'s module docstring) that writes the pending action (or
   checks the standing grant) *before* `fn` ever runs, and re-verifies it on the resumed call.
2. **Reads and clears** `ctx.deps.pending_approval` -- the hand-off `require_approval` just set --
   before calling `fn`, so the slot never carries a stale value into the next call.
3. **Calls `fn`** and records the outcome through `record_write_outcome`:
   - an exception propagates *after* `failed_to_execute` is recorded;
   - a `None` return (`fn`'s own "nothing to act on" sentinel, e.g. the target id did not resolve
     to a row) records `failed_to_execute` too, without raising -- this decorator reports a
     generic not-found text to the model instead of `fn`'s own wording, since a shared wrapper has
     no domain-specific phrasing to offer;
   - any other return value is the model-visible text, recorded as `executed`.

This is the only module that reads `AssistantDeps.pending_approval` -- `app/tools/approvals.py`
is the only one that sets it (`tests/test_run.py::test_pending_approval_is_read_only_here`
greps for both halves of that split).
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic_ai import Agent, RunContext

from app.tools.approvals import record_write_outcome, require_approval

if TYPE_CHECKING:
    from app.agents.assistant import AssistantDeps

_F = TypeVar("_F", bound=Callable[..., Awaitable[Any]])


def writing_tool(agent: Agent[AssistantDeps, Any]) -> Callable[[_F], _F]:
    """Registers `fn` on `agent` as a writing tool (module docstring). `fn` itself is
    `async def fn(ctx: RunContext[AssistantDeps], **args) -> str | None`: it does the one write
    and returns the model-visible text for success, or `None` for "nothing to act on" -- nothing
    else. Returns whatever `agent.tool(args_validator=require_approval)` itself returns (the
    registered function, pydantic-ai's own convention) so the decorated name stays callable."""

    def register(fn: _F) -> _F:
        @functools.wraps(fn)
        async def wrapped(ctx: RunContext[AssistantDeps], **kwargs: Any) -> str:
            approval = ctx.deps.pending_approval
            ctx.deps.pending_approval = None
            try:
                result = await fn(ctx, **kwargs)
            except Exception:
                await record_write_outcome(ctx.deps.ctx, approval, success=False)
                raise
            await record_write_outcome(ctx.deps.ctx, approval, success=result is not None)
            if result is None:
                return f"{ctx.tool_name}: nothing to act on -- no matching target was found."
            return result

        return agent.tool(args_validator=require_approval)(wrapped)

    return register


__all__ = ["writing_tool"]
