"""The default agent: PydanticAI with tools, context via deps.

- No model hard-wired: get_model() resolves it at runtime (provider abstraction, per tenant if
  needed). Tests override with TestModel/FunctionModel — no real model call.
- Tools are thin wrappers around app/tools/* that take the context from ctx.deps.
- LangGraph only once a flow becomes a state machine (checkpoints, human-in-the-loop) —
  then as its own module, with an ADR.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.result import StreamedRunResult

from app.context import RequestContext
from app.llm import get_model
from app.repositories.documents import DocumentHit
from app.run_limits import RunLimits, build_run_limits, run_deadline
from app.tools import documents as document_tools

SearchFn = Callable[[RequestContext, str, int], Awaitable[list[DocumentHit]]]


@dataclass
class AssistantDeps:
    ctx: RequestContext
    # Injectable so tests run without a database and embeddings (None = the real search).
    search: SearchFn | None = None
    model_name: str | None = None  # e.g. from tenants.settings["model"]

    def __post_init__(self) -> None:
        if self.search is None:
            self.search = document_tools.search_documents


INSTRUCTIONS = (
    "You are this application's assistant. Answer questions based on the user's documents. "
    "Use search_documents before stating facts, and name the titles of the documents you rely "
    "on. If nothing relevant is found, say so clearly."
)

assistant: Agent[AssistantDeps, str] = Agent(
    deps_type=AssistantDeps,
    instructions=INSTRUCTIONS,
    name="assistant",
    retries=2,
)


@assistant.tool
async def search_documents(
    ctx: RunContext[AssistantDeps], query: str, limit: int = 5
) -> list[DocumentHit]:
    """Searches the current user's documents semantically.

    Args:
        query: Search query in natural language.
        limit: Maximum number of hits (1–20).
    """
    assert ctx.deps.search is not None
    return await ctx.deps.search(ctx.deps.ctx, query, limit)


async def run_assistant(prompt: str, deps: AssistantDeps, limits: RunLimits | None = None) -> str:
    limits = limits or build_run_limits()
    async with run_deadline(limits):
        result = await assistant.run(
            prompt,
            deps=deps,
            model=get_model(deps.model_name),
            usage_limits=limits.usage_limits,
            metadata=deps.ctx.trace_attributes(),
        )
    return result.output


def stream_assistant(prompt: str, deps: AssistantDeps, limits: RunLimits | None = None):
    """Async context manager yielding a StreamedRunResult; use it via `async with` in routes.

    Does NOT itself enforce the run's wall-clock deadline: the deadline must bound the full
    open-and-consume lifecycle (opening the stream, then reading every delta from it), which
    means wrapping the caller's `async with ... as result: async for ...` block in
    `run_limits.run_deadline(limits)` — see `app/api/agents.py`'s `/assistant/stream` route.
    """
    limits = limits or build_run_limits()
    return assistant.run_stream(
        prompt,
        deps=deps,
        model=get_model(deps.model_name),
        usage_limits=limits.usage_limits,
        metadata=deps.ctx.trace_attributes(),
    )


__all__ = ["AssistantDeps", "StreamedRunResult", "assistant", "run_assistant", "stream_assistant"]
