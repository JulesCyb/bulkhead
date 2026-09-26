"""One agent run, prepared once (spec A3 / #93, #107): the one narrative of what a run is.

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
   are the real ones unless a caller injects its own.

Suspension is not checked here (#106, ADR-0010): context resolution already refused a suspended
tenant's record, and `tenant_session()` refuses one for any session a tool opens.

**Execution** is one of the prepared run's methods; which agent it binds is decided by the method,
never by a flag (ADR-0007):

- `answer(prompt)` -- the one-shot JSON response (`/agents/assistant/run`).
- `stream_text(prompt)` -- the one-shot text stream (`/agents/assistant/stream`).

Both bind `one_shot_assistant`, the reading-only agent: no tool with an approval validator is
registered on it, so neither can ever propose a write that has no conversation to be approved in.
Both wrap the *full* open-and-consume lifecycle -- for a stream, every delta read, not only the
call that opens it -- in the run's deadline (`run_deadline`) and span attributes
(`tenant_span_attributes`), and pass the usage ceiling, the trace metadata, and the tracing
capabilities to the run. A route wraps nothing.

A third method, `chat(adapter)` for the Vercel AI SDK protocol -- the only one that will bind the
writing-capable `chat_assistant`, load the server-held history, resolve incoming approval
decisions, persist on completion and decouple from the client (ADR-0006, ADR-0007) -- is added by
#108; until then `app/api/chat.py` takes only its model from here.

**Tests** inject the model and the tool functions as `prepare_run` arguments. A test that drives a
route (which calls `prepare_run(ctx)` with no collaborators) installs them with
`set_run_collaborators_for_tests` instead -- the one test hook, mirroring
`app.token_verifier.set_default_adapter_for_tests`; `tests/conftest.py`'s `prepared_run` and
`route_run` fixtures wrap both. Nothing patches a module attribute.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from pydantic_ai.models import Model

from app.agents.assistant import (
    AssistantDeps,
    LoadHistoryFn,
    SaveRunFn,
    SearchFn,
    one_shot_assistant,
)
from app.context import RequestContext
from app.llm import resolve_tenant_chat_model
from app.observability import (
    TenantTracingSelection,
    instrumentation_capabilities,
    resolve_tenant_tracing,
    tenant_span_attributes,
)
from app.residency import ResidencyUnresolved
from app.run_limits import RunLimits, build_run_limits, run_deadline
from app.tenant_record import TenantRecord

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
    (`AssistantDeps.conversation_id`); `None` for a one-shot run. `settings` is passed through to
    the model resolver and the run limit (`None` = each one's own `get_settings()`). Every other
    keyword is a collaborator: an explicit argument wins, then one installed with
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
    "ModelResolver",
    "PreparedRun",
    "RunCollaborators",
    "prepare_run",
    "set_run_collaborators_for_tests",
]
