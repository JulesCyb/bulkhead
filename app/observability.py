"""Tracing for agent runs (Langfuse via OpenTelemetry) — ADR-0008, Spec 8 / #62.

Content-free by default, per-tenant opt-in, per-residency sink:

- Every span an agent run produces carries `tenant_id`/`identity_id`/`request_id` as flat span
  attributes (`_TenantAttributeSpanProcessor` below, driven by a contextvar set for the run's
  whole duration) — not only as `metadata` on the root span.
- Prompts, tool arguments, and document text (`include_content`) are captured only when the
  calling tenant's own `tenants.settings["content_tracing_opt_in"]` is true
  (`app.tenant_settings.TenantSettings`, carried on the tenant record) — resolved per run, never
  a process-wide toggle.
- The trace sink a run's spans reach is resolved from the tenant's residency exactly like the
  model and embedding routes: one `TracerProvider` per residency
  (`settings.residency_allow_list.route_for(residency).trace_sink_host`,
  `app.residency.ResidencyAllowList`), so two tenants in different
  residencies can never share an exporter, and a tenant whose residency can't be resolved simply
  runs untraced rather than falling back to someone else's sink.
- The OTLP endpoint and the Langfuse Basic-auth header are set directly as constructor arguments
  on the exporter this module builds (`OTLPSpanExporter(endpoint=..., headers=...)`) — never
  through the `OTEL_EXPORTER_OTLP_ENDPOINT`/`OTEL_EXPORTER_OTLP_HEADERS` environment variables, so
  a platform-level variable set outside the application's control can no longer silently redirect
  where content-bearing spans go.
- `setup_observability()` raises `ObservabilityInitializationError` if Langfuse credentials are
  configured but a `TracerProvider`/exporter can't be built — a startup error, never a warning
  that lets the process serve traffic untraced. A deployment with no credentials configured at
  all starts fine and simply never traces (`instrumentation_capabilities()` then always
  returns `[]`).

Per-run wiring lives at the call sites that actually run an agent (`app/agents/assistant.py`,
`app/api/agents.py`, `app/api/chat.py`): each takes the calling tenant's residency and content
opt-in from the tenant record the request's context already carries (`resolve_tenant_tracing`,
below -- no database read of its own, #105; the model is resolved from the very same record, so
tracing and the model can never disagree about the tenant's residency), builds this run's
`capabilities=` from `instrumentation_capabilities()`, and wraps the run in
`tenant_span_attributes(...)`. Never
`Agent.instrument_all()` or a mutated agent-level `.instrument` — those are process-wide, and
would let one tenant's residency/opt-in apply to a concurrent request for a different tenant.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from uuid import UUID

from opentelemetry.context import Context as OtelContext
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import Span
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.models.instrumented import InstrumentationSettings

from app.config import Settings
from app.tenant_record import TenantRecord

log = logging.getLogger(__name__)


class ObservabilityInitializationError(RuntimeError):
    """Langfuse credentials are configured but a `TracerProvider`/exporter could not be built.

    Raised from `setup_observability()` so a broken tracing setup fails startup loudly instead of
    quietly leaving the process tracing nothing (ADR-0008) — never caught to fall back to
    untraced operation; a deployment that wants to run untraced does so by not configuring
    Langfuse credentials at all, not by tracing failing silently.
    """


# --- Flat tenant/user attributes on every span, not only the root span's metadata -------------

# Set for the duration of one agent run (`tenant_span_attributes`, below), read by
# `_TenantAttributeSpanProcessor.on_start` so every span the run produces — the root agent-run
# span, every model-request span, every tool-call span — carries the same flat attributes.
# Contextvars propagate through `await` within one asyncio task and into any task spawned from
# it (a fresh copy of the current context), which is exactly the lifetime of one run, streamed or
# not — as long as the `with` block covers the run's full open-and-consume lifecycle.
_trace_attributes: ContextVar[dict[str, str] | None] = ContextVar("_trace_attributes", default=None)


@contextmanager
def tenant_span_attributes(attributes: dict[str, str]) -> Iterator[None]:
    """Attaches `attributes` (`RequestContext.trace_attributes()`) to every span created while
    the `with` block is open. For a streamed run, wrap the *entire* open-and-consume loop, not
    just the call that starts the stream — a plain async generator resumed later, outside this
    block, would otherwise pick up whatever contextvar value is ambient at resume time."""
    token = _trace_attributes.set(dict(attributes))
    try:
        yield
    finally:
        _trace_attributes.reset(token)


class _TenantAttributeSpanProcessor(SpanProcessor):
    """Stamps whatever `tenant_span_attributes()` currently has set onto every span as it
    starts — never from a `metadata` blob a downstream query would have to parse. Purely
    additive: it never exports anything itself, so it is added to every residency's
    `TracerProvider` ahead of that residency's own exporting processor (`build_tracer_provider`,
    below): by the time a span reaches the exporter, these attributes are already on it.
    """

    def on_start(self, span: Span, parent_context: OtelContext | None = None) -> None:
        attributes = _trace_attributes.get()
        if attributes:
            span.set_attributes(attributes)

    def on_end(self, span: ReadableSpan) -> None:  # pragma: no cover - nothing to do
        return None

    def shutdown(self) -> None:  # pragma: no cover - nothing to do
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:  # pragma: no cover - no-op
        return True


def build_tracer_provider(
    exporter: SpanExporter, *, processor_cls: type[SpanProcessor] = BatchSpanProcessor
) -> TracerProvider:
    """A `TracerProvider` wired with the tenant-attribute stamping above, plus `exporter` wrapped
    in `processor_cls` — `BatchSpanProcessor` (the production default) batches asynchronously;
    tests typically pass `SimpleSpanProcessor` so a captured span is visible the instant the span
    ends, with no batching delay to wait out.
    """
    provider = TracerProvider()
    provider.add_span_processor(_TenantAttributeSpanProcessor())
    provider.add_span_processor(processor_cls(exporter))
    return provider


# --- One TracerProvider (and, for test/introspection use, one exporter) per residency ----------

# Populated by `setup_observability()`; cleared to `{}` when tracing isn't configured at all.
# `instrumentation_capabilities()` is the only production reader. Tests inject a fake provider
# per residency via `set_tracer_provider_for_residency` without going through the real OTLP
# exporter `setup_observability()` builds.
_residency_tracer_providers: dict[str, TracerProvider] = {}

# Exporters `setup_observability()` itself built, kept only so a test can read back exactly what
# endpoint was configured (`configured_trace_endpoint`) — proving the destination came from
# the residency allow-list, not from an `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable. No
# production code reads this dict.
_residency_exporters: dict[str, OTLPSpanExporter] = {}


def setup_observability(settings: Settings) -> bool:
    """Builds one `TracerProvider` per residency in `settings.residency_allow_list`, each pointed
    at that residency's own `trace_sink_host` (ADR-0008) — called once from the ASGI lifespan
    (`app/main.py`).

    Returns `False`, with tracing left off, when no Langfuse credentials are configured at all —
    a deployment with nothing configured still starts and serves requests, untraced. Raises
    `ObservabilityInitializationError` if credentials *are* configured but building a provider
    fails: a broken tracing setup must never let the process quietly start untraced instead of
    failing at startup.
    """
    _residency_tracer_providers.clear()
    _residency_exporters.clear()
    if not (
        settings.langfuse_host and settings.langfuse_public_key and settings.langfuse_secret_key
    ):
        return False
    try:
        secret_key = settings.langfuse_secret_key.get_secret_value()
        auth = base64.b64encode(f"{settings.langfuse_public_key}:{secret_key}".encode()).decode()
        auth_header = f"Basic {auth}"
        providers: dict[str, TracerProvider] = {}
        exporters: dict[str, OTLPSpanExporter] = {}
        allow_list = settings.residency_allow_list
        assert allow_list is not None  # set by Settings construction
        for residency in allow_list.residencies:
            route = allow_list.route_for(residency)
            exporter = OTLPSpanExporter(
                endpoint=f"https://{route.trace_sink_host}/api/public/otel",
                headers={"Authorization": auth_header},
            )
            providers[residency] = build_tracer_provider(exporter)
            exporters[residency] = exporter
    except Exception as exc:
        raise ObservabilityInitializationError(
            "Langfuse is configured (LANGFUSE_HOST/LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY) but "
            "tracing failed to initialize -- refusing to start untraced with a broken tracing "
            "setup."
        ) from exc
    _residency_tracer_providers.update(providers)
    _residency_exporters.update(exporters)
    return True


def set_tracer_provider_for_residency(residency: str, provider: TracerProvider) -> None:
    """Test-only: injects a fake `TracerProvider` (typically `build_tracer_provider` over an
    `InMemorySpanExporter`) for one residency, without going through `setup_observability()`'s
    real OTLP exporter or requiring real Langfuse credentials."""
    _residency_tracer_providers[residency] = provider


def reset_tracer_providers() -> None:
    """Test-only: clears every configured tracer provider (and exporter) between test cases."""
    _residency_tracer_providers.clear()
    _residency_exporters.clear()


def is_tracing_configured() -> bool:
    """True once at least one residency has a `TracerProvider` — either a real one from
    `setup_observability()` or a fake one a test injected via
    `set_tracer_provider_for_residency`. Introspection only: `instrumentation_capabilities()`
    already returns `[]` when nothing is configured, and resolving a tenant's tracing costs no
    database round trip either way (#105)."""
    return bool(_residency_tracer_providers)


def configured_trace_endpoint(residency: str) -> str | None:
    """Test-only introspection: the exact OTLP endpoint URL `setup_observability()` configured
    for `residency`'s exporter — proves that endpoint came from the residency allow-list, set
    directly on the exporter, never from `OTEL_EXPORTER_OTLP_ENDPOINT`."""
    exporter = _residency_exporters.get(residency)
    return exporter._endpoint if exporter is not None else None  # noqa: SLF001


def instrumentation_capabilities(
    residency: str | None, content_tracing_opt_in: bool
) -> list[Instrumentation]:
    """The `capabilities=` list a `pydantic_ai.Agent.run`/`run_stream`/
    `VercelAIAdapter.dispatch_request` call for this run should use.

    **The one deliberate fail-open in residency handling (ADR-0008, spec #92):** returns `[]`
    (this run goes untraced) whenever there is no trace sink for the tenant's residency -- the
    residency is unresolved (`None`, e.g. a record with none recorded or a context with no record
    at all), unknown to the allow-list, or tracing is not configured for any residency. Unlike the
    model and embedding paths, which must fail the request closed (`app.residency.
    ResidencyUnresolved`), tracing is not content-bearing on its own: an untraced run can neither
    cross a residency boundary nor leak content, whereas refusing the request would make a
    tracing-only gap user-facing. What it never does is fall back to another residency's sink.

    `content_tracing_opt_in` maps straight onto `InstrumentationSettings.include_content`:
    prompts and tool results are captured only when the tenant has explicitly opted in, never by
    default (ADR-0008). A fresh `Instrumentation`/`InstrumentationSettings` is built on every
    call — never a shared instance held across runs or agents — so one run's settings can never
    leak into a concurrent run for a different tenant.
    """
    provider = _residency_tracer_providers.get(residency) if residency else None
    if provider is None:  # the one fail-open branch -- see the docstring
        return []
    instrumentation_settings = InstrumentationSettings(
        tracer_provider=provider, include_content=content_tracing_opt_in
    )
    return [Instrumentation(settings=instrumentation_settings)]


# --- Per-request resolution of residency + content opt-in, from the tenant record -------------


@dataclass(frozen=True, slots=True)
class TenantTracingSelection:
    """What one request needs to trace its agent run correctly: the tenant's residency (`None` if
    unresolved — the run then simply goes untraced, see `instrumentation_capabilities`) and
    whether that tenant has opted the whole tenant into content-bearing tracing."""

    residency: str | None
    content_tracing_opt_in: bool


def resolve_tenant_tracing(record: TenantRecord | None) -> TenantTracingSelection:
    """The tracing selection for a tenant, as a function of its record (#105): its residency
    (`record.residency`, operator-owned, ADR-0008) and its content-tracing opt-in
    (`record.settings.content_tracing_opt_in`, tenant-owned) -- the same record, read once per
    request, that the model and embedding routes resolve from, so there is no second residency
    read that could disagree with theirs.

    Never raises: a record without a residency is carried through as `None`, and a context with
    no record at all (a job, a test) gets the untraced, content-off selection -- both end in
    `instrumentation_capabilities`'s one fail-open branch (untraced), never a fallback sink.
    """
    if record is None:
        return TenantTracingSelection(residency=None, content_tracing_opt_in=False)
    return TenantTracingSelection(
        residency=record.residency,
        content_tracing_opt_in=record.settings.content_tracing_opt_in,
    )


async def delete_tenant_traces(tenant_id: UUID) -> None:
    """Placeholder per-tenant trace-deletion capability (ADR-0010, Spec 9 / #72).

    Spec 8's own scope note (docs describing this spec's dependency on Spec 8) says erasure "calls
    a per-tenant trace-deletion capability that Spec 8 must expose" -- no such capability exists
    yet against any configured tracing backend (Langfuse has no documented per-tenant hard-delete
    API as of ADR-0008's research), and building one is explicitly Spec 8's job, not this one's.

    This stub is the interface `app.operator.erase.erase_tenant` calls through so that a real
    implementation, once Spec 8 delivers one, is a one-function replacement here rather than a
    change to the erase module or any of its call sites. It never raises and never reaches a real
    tracing backend; a test replaces it with a fake via `erase_tenant`'s `trace_deleter=` seam to
    observe that erasure actually requested it.
    """
    return None


__all__ = [
    "ObservabilityInitializationError",
    "TenantTracingSelection",
    "build_tracer_provider",
    "configured_trace_endpoint",
    "delete_tenant_traces",
    "instrumentation_capabilities",
    "is_tracing_configured",
    "reset_tracer_providers",
    "resolve_tenant_tracing",
    "set_tracer_provider_for_residency",
    "setup_observability",
    "tenant_span_attributes",
]
