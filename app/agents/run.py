"""One agent run, prepared once (spec A3 / #93, #107): the one narrative of what a run is.

The reading/writing split (ADR-0007) lives here and in `app/agents/writing_tools.py` next to it:
`AssistantDeps` (built only by `prepare_run`, below) is internal to a run, and its
`pending_approval` slot is read and cleared by exactly one thing -- the `writing_tool` decorator
this module re-exports (#109) -- never by a route or a tool body directly. A writing tool applies
`@writing_tool(chat_assistant)` (`app/agents/assistant.py`'s `rename_document` is the worked
example) instead of copying that read/clear/execute/record sequence by hand.

A route (or any future entry point, e.g. a jobs API) calls `prepare_run(ctx)` with the context
it already resolved -- carrying the tenant record read once at context resolution (#104/#105) --
and gets a `PreparedRun` back. Preparation does, in this order and exactly once per run:

1. **Model** (ADR-0008, ADR-0009): resolved from `ctx.tenant_record` -- its residency, its own
   `model` setting (else the deployment default) validated against that residency's allow-list,
   its gateway credential alias -- through `app.llm.resolve_tenant_chat_model`. A context without
   a record (a job, a test) fails closed with `ResidencyUnresolved` before anything else happens:
   never a control-plane read of its own, never the deployment's residency. The resolver's own
   `ResidencyUnresolved`/`ModelNotAllowedForResidency`/`GatewayCredentialUnavailable` propagate
   unchanged; `app.agents.run_errors` maps them for every transport.
2. **Run limit** (ADR-0009, `CONTEXT.md` "Run limit"): `build_run_limits()` once -- the usage
   ceiling passed to the run and the wall-clock deadline wrapped around it.
3. **Tracing** (ADR-0008): the tenant's residency and content-tracing opt-in from the same record
   (`resolve_tenant_tracing`), the `capabilities=` list for that residency's own sink
   (`instrumentation_capabilities`, a fresh instance per run -- untraced if the residency has no
   sink, never another residency's), and the flat span attributes from the context
   (`RequestContext.trace_attributes()`: tenant, identity, request id).
4. **Tool dependencies**: the agent dependency object (`AssistantDeps`) the tools read, built here
   and nowhere else -- no route constructs it. Its tool functions (search, history, persistence)
   are the real ones unless a caller injects its own. For a chat run it also carries the bare
   conversation id (`conversation_id=`) a writing tool's approval is scoped to.

Suspension is not checked here (#106, ADR-0010): context resolution already refused a suspended
tenant's record, and `tenant_session()` refuses one for any session a tool opens.

**Execution** is one of the prepared run's methods; which agent it binds is decided by the method,
never by a flag (ADR-0007):

- `answer(prompt)` -- the one-shot JSON response (`/agents/assistant/run`).
- `stream_text(prompt)` -- the one-shot text stream (`/agents/assistant/stream`).
- `chat(adapter)` -- the Vercel AI SDK chat stream (`/api/chat`).

`answer` and `stream_text` bind `one_shot_assistant`, the reading-only agent: no tool with an
approval validator is registered on it, so neither can ever propose a write that has no
conversation to be approved in. Only `chat` binds the writing-capable `chat_assistant` -- whatever
agent the adapter it is handed was built with. Every method wraps the *full* open-and-consume
lifecycle -- for a stream, every chunk read, not only the call that opens it -- in the run's
deadline (`run_deadline`) and span attributes (`tenant_span_attributes`), and passes the usage
ceiling, the trace metadata, and the tracing capabilities to the run. A route wraps nothing.

**A chat run** (`chat(adapter)`, #108) is the one surface with a conversation, so it owns, in
this order:

1. **Member-authored input only** (ADR-0006, Spec 4 / #33). The only client input a chat run
   trusts is the newest message's member-authored content. The adapter's own `messages` (what
   `VercelAIAdapter.dispatch_request()` would feed the agent) is built from *every* message in
   the body, only stripped of system prompts and disallowed file references
   (`UIAdapter.sanitize_messages`) -- never of earlier turns or of a forged assistant/tool part
   riding along on the newest one. So `chat` overrides it with just the newest, member-authored
   turn (`_member_authored_messages`) before running.
2. **Server-held history** (ADR-0006, #33): loaded for the adapter's conversation id through the
   run's `load_history` (the real `ConversationsRepository`, scoped to the tenant and to the
   member who started the conversation), never taken from the body.
3. **Approval decisions, before the run** (ADR-0007, #40): a resumed request may carry the
   member's approve/refuse decision for a writing tool's earlier deferred call
   (`VercelAIAdapter.deferred_tool_results`, extracted from the raw body regardless of the
   `messages` override above). A refusal is resolved by pydantic-ai substituting `ToolDenied`
   directly -- the tool's own `args_validator` never runs for a denied call -- so this is the only
   place a refusal's audit milestone can be recorded; an approval is resolved here too, before the
   run starts, so the tool's own execution-time re-verification always has an `approved` (not
   merely `pending`) record to check against (`app.tools.approvals.resolve_incoming_decisions`).
4. **Tenant-scoped conversation id** (ADR-0006, #34): the id the agent run itself carries (and
   tracing reports) is `"{tenant_id}:{conversation_id}"`, never the client's bare id -- tracing has
   no Row-Level Security to fall back on if two tenants' clients ever pick the same id. The bare
   id stays what history, persistence, and a pending action are keyed by.
5. **Persistence on completion, decoupled from the client** (ADR-0006, #34): the run's newly
   produced messages are written back through the run's `save_run` (the real
   `ConversationsRepository`, in a session of its own) from the adapter's `on_complete` hook, which
   fires when the run itself finishes successfully -- a run that raises persists nothing. But a
   `StreamingResponse.body_iterator` a client stops reading from is a generator that simply never
   advances, so persistence would still depend on the client finishing the read unless something
   keeps driving the stream: `_decouple_from_client` drains it into a queue from a background task
   the ASGI server's own consumption cannot stall.
6. **Deadline and span attributes around consumption**: `adapter.run_stream()` only builds an
   async generator that does its work once iterated, so the deadline and span attributes wrap the
   encoded stream's consumption (`_bounded_by_deadline`); on timeout the stream ends with one
   Vercel AI SDK `error` chunk -- the same shape the adapter itself emits for an in-run
   `UsageLimitExceeded` -- never a raw exception or a connection that just stops.

The chat route keeps only transport (`app/api/chat.py`): the body-size guard, parsing the body
into the adapter (`chat_adapter_from_request`, which fixes the protocol version the approval
round-trip needs), preparing the run for the adapter's conversation, and the error mapping.

**Tests** inject the model and the tool functions as `prepare_run` arguments. A test that drives a
route (which calls `prepare_run(ctx)` with no collaborators) installs them with
`set_run_collaborators_for_tests` instead -- the one test hook, mirroring
`app.token_verifier.set_default_adapter_for_tests`; `tests/conftest.py`'s `prepared_run` and
`route_run` fixtures wrap both. Nothing patches a module attribute.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Coroutine, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from pydantic_ai import DeferredToolRequests
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from pydantic_ai.ui.vercel_ai.request_types import FileUIPart, TextUIPart, UIMessage
from starlette.requests import Request
from starlette.responses import StreamingResponse

from app.agents.assistant import (
    AssistantDeps,
    LoadHistoryFn,
    SaveRunFn,
    SearchFn,
    chat_assistant,
    one_shot_assistant,
)
from app.agents.writing_tools import writing_tool
from app.context import RequestContext
from app.llm import resolve_tenant_chat_model
from app.observability import (
    TenantTracingSelection,
    instrumentation_capabilities,
    resolve_tenant_tracing,
    tenant_span_attributes,
)
from app.residency import ResidencyUnresolved
from app.run_limits import RunDeadlineExceeded, RunLimits, build_run_limits, run_deadline
from app.tenant_record import TenantRecord
from app.tools.approvals import resolve_incoming_decisions

if TYPE_CHECKING:
    from pydantic_ai.capabilities.instrumentation import Instrumentation

    from app.config import Settings


class ModelResolver(Protocol):
    """Resolves the chat model for one tenant record -- `app.llm.resolve_tenant_chat_model`'s
    shape, which is the default."""

    def __call__(self, record: TenantRecord, *, settings: Settings | None = None) -> Model: ...


@dataclass(frozen=True, slots=True)
class RunCollaborators:
    """The replaceable collaborators of a run. `None` in any field means "the real one"."""

    model_resolver: ModelResolver | None = None
    search: SearchFn | None = None
    load_history: LoadHistoryFn | None = None
    save_run: SaveRunFn | None = None


# Test-only override installed via `set_run_collaborators_for_tests` below -- `None` means "the
# real collaborators". Read only by `prepare_run`; production code never sets it.
_test_collaborators: RunCollaborators | None = None


def set_run_collaborators_for_tests(collaborators: RunCollaborators | None) -> None:
    """Test-only hook (#107): installs `collaborators` as what `prepare_run` uses for every
    collaborator its caller does not pass explicitly -- the seam a test driving a route over ASGI
    uses (the route calls `prepare_run(ctx)` with none), instead of patching a module attribute.
    A field left `None` keeps the real collaborator. Call with `None` to restore all of them."""
    global _test_collaborators
    _test_collaborators = collaborators


# --- The chat protocol: the adapter a chat run executes against ----------------------------------

ChatAdapter = VercelAIAdapter[AssistantDeps, str | DeferredToolRequests]


async def chat_adapter_from_request(request: Request) -> ChatAdapter:
    """Parses `request`'s body into the Vercel AI SDK adapter a chat run executes against.

    Fixes the protocol version (`sdk_version=6`, the first that carries tool approval
    round-trips; `VercelAIAdapter.deferred_tool_results` is always empty below it) so no caller
    can drop the approval half of ADR-0007 by building the adapter differently. Raises
    `pydantic.ValidationError` for a malformed body -- the route maps that to its 422."""
    return await VercelAIAdapter.from_request(request, agent=chat_assistant, sdk_version=6)


def _member_authored_messages(messages: list[UIMessage]) -> list[ModelMessage]:
    """The only part of the request body a chat run trusts as new input (module docstring).

    Takes the newest message and, only if it is the member's own turn (`role == "user"`),
    keeps just the part types a member can actually author -- text and file parts -- before
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


def _tenant_scoped_conversation_id(ctx: RequestContext, conversation_id: str) -> str | None:
    """The conversation id the agent run itself carries (and tracing reports): combined with the
    tenant id, never the client's bare id alone (module docstring). `None` without a conversation
    id, so pydantic-ai generates one of its own rather than a tenant-only id every such run
    would share."""
    return f"{ctx.tenant_id}:{conversation_id}" if conversation_id else None


async def _bounded_by_deadline(
    source: AsyncIterator[str], limits: RunLimits, attributes: dict[str, str]
) -> AsyncIterator[str]:
    """Wraps the adapter's already-encoded chat stream in the run's wall-clock deadline and span
    attributes for as long as it is actually consumed -- the only point that covers the plain
    async generator's full resumed lifetime (module docstring). On timeout, emits one Vercel AI
    SDK `error` chunk -- the same shape pydantic-ai's own adapter emits for an in-run exception
    like `UsageLimitExceeded` -- so the client sees one mapped error, never a raw exception or a
    connection that just stops.
    """
    try:
        async with run_deadline(limits):
            with tenant_span_attributes(attributes):
                async for chunk in source:
                    yield chunk
    except RunDeadlineExceeded as exc:
        payload = json.dumps({"type": "error", "errorText": str(exc)}, separators=(",", ":"))
        yield f"data: {payload}\n\n"


# Fire-and-forget background tasks (the stream-draining task started by `_decouple_from_client`)
# are only weakly referenced by the event loop once nothing else holds them -- keeping a strong
# reference here until each finishes is what stops asyncio from garbage-collecting one mid-flight.
_background_tasks: set[asyncio.Task[None]] = set()


def _spawn_background(coro: Coroutine[Any, Any, None]) -> None:
    task = asyncio.ensure_future(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _drain_into_queue(source: AsyncIterator[str], queue: asyncio.Queue[str | None]) -> None:
    """Pulls every chunk out of `source` -- driving it to exhaustion, including whatever
    `on_complete` callback fires on its last item -- regardless of whether anything is still
    reading the other end of `queue`. Errors already surface as an encoded `error` chunk
    upstream (`_bounded_by_deadline`, the adapter's own `on_error`), so nothing here is expected
    to raise; the `except` is a last-resort guard against an unretrieved-task-exception warning
    if one somehow does, not a place that swallows a persistence failure.
    """
    try:
        async for chunk in source:
            await queue.put(chunk)
    except Exception:
        pass
    finally:
        await queue.put(None)


async def _queue_iterator(queue: asyncio.Queue[str | None]) -> AsyncIterator[str]:
    while (item := await queue.get()) is not None:
        yield item


def _decouple_from_client(source: AsyncIterator[str]) -> AsyncIterator[str]:
    """Returns an iterator fed from a background task that drains `source` on its own, so a
    client that stops reading the response never stalls `source` itself (ADR-0006, #34) -- in
    particular, the `on_complete` persistence hook attached to `source` still runs to completion.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    _spawn_background(_drain_into_queue(source, queue))
    return _queue_iterator(queue)


@dataclass(frozen=True, slots=True)
class PreparedRun:
    """Everything one run needs, assembled once by `prepare_run` -- see the module docstring.

    One prepared run is one run: call exactly one execution method, once."""

    ctx: RequestContext
    model: Model
    limits: RunLimits
    tracing: TenantTracingSelection
    _capabilities: Sequence[Instrumentation]
    # Internal: the dependency object the tools read. Never constructed or read by a route.
    _deps: AssistantDeps

    async def answer(self, prompt: str) -> str:
        """Runs the reading-only agent to completion and returns its answer."""
        attributes = self.ctx.trace_attributes()
        async with run_deadline(self.limits):
            with tenant_span_attributes(attributes):
                result = await one_shot_assistant.run(
                    prompt,
                    deps=self._deps,
                    model=self.model,
                    usage_limits=self.limits.usage_limits,
                    metadata=attributes,
                    capabilities=self._capabilities,
                )
        return result.output

    async def stream_text(self, prompt: str) -> AsyncIterator[str]:
        """Runs the reading-only agent as a stream, yielding its text deltas. The deadline and
        span attributes bound opening the stream *and* reading every delta from it."""
        attributes = self.ctx.trace_attributes()
        async with run_deadline(self.limits):
            with tenant_span_attributes(attributes):
                async with one_shot_assistant.run_stream(
                    prompt,
                    deps=self._deps,
                    model=self.model,
                    usage_limits=self.limits.usage_limits,
                    metadata=attributes,
                    capabilities=self._capabilities,
                ) as result:
                    async for delta in result.stream_text(delta=True):
                        yield delta

    async def chat(self, adapter: ChatAdapter) -> StreamingResponse:
        """Runs the writing-capable chat agent against `adapter` -- the request body the route
        parsed (`chat_adapter_from_request`) -- and returns the streaming response. Everything a
        chat run owns (module docstring, "A chat run") happens here, in that order.

        The run must have been prepared for the adapter's own conversation
        (`prepare_run(ctx, conversation_id=adapter.conversation_id)`); a mismatch is a caller bug
        and raises `ValueError` before anything is loaded or run."""
        conversation_id = self._deps.conversation_id or ""
        if conversation_id != (adapter.conversation_id or ""):
            raise ValueError("this run was prepared for a different conversation than the adapter")
        # Which agent runs is decided here, by the method -- never by how the adapter was built.
        adapter.agent = chat_assistant
        # Overrides the adapter's own `messages` cached property (normally every message in the
        # body) with just the newest, member-authored turn.
        adapter.messages = _member_authored_messages(adapter.run_input.messages)

        load_history, save_run = self._deps.load_history, self._deps.save_run
        assert load_history is not None and save_run is not None  # AssistantDeps fills both
        history = await load_history(self.ctx, conversation_id)

        deferred_tool_results = adapter.deferred_tool_results
        if deferred_tool_results is not None:
            await resolve_incoming_decisions(
                self.ctx, conversation_id=conversation_id, decisions=deferred_tool_results.approvals
            )

        async def _persist_new_messages(result: AgentRunResult[Any]) -> None:
            # `on_complete` only fires when the run finishes successfully -- a run that raises
            # before completing never reaches here, so nothing is persisted for it.
            await save_run(self.ctx, conversation_id, result.new_messages())

        attributes = self.ctx.trace_attributes()
        response = adapter.streaming_response(
            adapter.run_stream(
                message_history=history,
                deps=self._deps,
                model=self.model,
                usage_limits=self.limits.usage_limits,
                metadata=attributes,
                capabilities=self._capabilities,
                conversation_id=_tenant_scoped_conversation_id(self.ctx, conversation_id),
                on_complete=_persist_new_messages,
            )
        )
        response.body_iterator = _decouple_from_client(
            _bounded_by_deadline(response.body_iterator, self.limits, attributes)
        )
        return response


async def prepare_run(
    ctx: RequestContext,
    *,
    conversation_id: str | None = None,
    settings: Settings | None = None,
    model_resolver: ModelResolver | None = None,
    search: SearchFn | None = None,
    load_history: LoadHistoryFn | None = None,
    save_run: SaveRunFn | None = None,
) -> PreparedRun:
    """Prepares one run for `ctx` -- see the module docstring for what, in which order.

    `conversation_id` is the bare (not tenant-prefixed) id a writing tool's approval is scoped to
    (`AssistantDeps.conversation_id`) -- for a chat run, the adapter's own (`PreparedRun.chat`
    checks they agree); `None` for a one-shot run. `settings` is passed through to the model
    resolver and the run limit (`None` = each one's own `get_settings()`). Every other keyword is
    a collaborator: an explicit argument wins, then one installed with
    `set_run_collaborators_for_tests`, then the real one.

    Raises `ResidencyUnresolved` for a context without a tenant record, and whatever the model
    resolver raises (`app.agents.run_errors.ROUTING_ERRORS`) -- all before any model call.
    """
    record = ctx.tenant_record
    if record is None:
        raise ResidencyUnresolved(
            f"context for tenant {ctx.tenant_id} carries no tenant record to resolve a model from"
        )
    installed = _test_collaborators or RunCollaborators()
    resolver = model_resolver or installed.model_resolver or resolve_tenant_chat_model
    model = resolver(record, settings=settings)
    limits = build_run_limits(settings)
    tracing = resolve_tenant_tracing(record)
    deps = AssistantDeps(
        ctx=ctx,
        search=search or installed.search,
        load_history=load_history or installed.load_history,
        save_run=save_run or installed.save_run,
        conversation_id=conversation_id,
        residency=tracing.residency,
        content_tracing_opt_in=tracing.content_tracing_opt_in,
    )
    return PreparedRun(
        ctx=ctx,
        model=model,
        limits=limits,
        tracing=tracing,
        _capabilities=instrumentation_capabilities(
            tracing.residency, tracing.content_tracing_opt_in
        ),
        _deps=deps,
    )


__all__ = [
    "ChatAdapter",
    "ModelResolver",
    "PreparedRun",
    "RunCollaborators",
    "chat_adapter_from_request",
    "prepare_run",
    "set_run_collaborators_for_tests",
    "writing_tool",
]
