"""Unit tests for `app/observability.py` (Spec 8 / #62, ADR-0008): content-free-by-default
tracing, per-tenant opt-in, per-residency trace sinks, and a fail-loud startup guard — exercised
directly against the module (real `Settings`, tenant records constructed directly -- #105), no
real database and no real network call. The full ASGI-level, end-to-end behavior (spans an
actual agent run produces) lives in `tests/test_content_tracing.py`.
"""

from __future__ import annotations

import uuid

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app import observability
from app.config import Settings
from app.residency import ResidencyAllowList
from app.tenant_record import TenantRecord
from app.tenant_settings import TenantSettings


@pytest.fixture(autouse=True)
def _reset():
    observability.reset_tracer_providers()
    yield
    observability.reset_tracer_providers()


def _configured_settings(**overrides) -> Settings:
    return Settings(
        database_url="postgresql+asyncpg://app:app@localhost:5432/app",
        langfuse_host="https://ignored-by-design.example.com",
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        **overrides,
    )


# --- setup_observability: fail-closed / fail-loud startup behavior -----------------------------


def test_no_credentials_at_all_leaves_tracing_off_and_the_process_still_starts():
    settings = Settings(database_url="postgresql+asyncpg://app:app@localhost:5432/app")
    assert observability.setup_observability(settings) is False
    assert observability.is_tracing_configured() is False
    assert observability.instrumentation_capabilities("eu", True) == []


def test_credentials_configured_builds_one_tracer_provider_per_residency():
    assert observability.setup_observability(_configured_settings()) is True
    assert observability.is_tracing_configured() is True
    for residency in ResidencyAllowList.load().residencies:
        assert observability.instrumentation_capabilities(residency, False) != []


def test_broken_tracing_setup_raises_at_startup_instead_of_starting_untraced(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(observability, "OTLPSpanExporter", _boom)

    with pytest.raises(observability.ObservabilityInitializationError):
        observability.setup_observability(_configured_settings())
    # A failed setup must not leave a half-built, partially-working configuration behind.
    assert observability.is_tracing_configured() is False


def test_otlp_endpoint_env_var_never_affects_the_configured_destination(monkeypatch):
    """AC: pre-setting the OTLP endpoint environment variable before tracing initializes has no
    effect on where the exporter actually sends spans."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://attacker.example.com/v1/traces")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "Authorization=Bearer stolen-token")

    observability.setup_observability(_configured_settings())

    allow_list = ResidencyAllowList.load()
    for residency in allow_list.residencies:
        route = allow_list.route_for(residency)
        endpoint = observability.configured_trace_endpoint(residency)
        assert endpoint == f"https://{route.trace_sink_host}/api/public/otel"
        assert "attacker.example.com" not in endpoint


# --- instrumentation_capabilities: residency resolution, content opt-in ------------------------


def test_no_capabilities_when_residency_is_unresolved():
    observability.setup_observability(_configured_settings())
    assert observability.instrumentation_capabilities(None, True) == []


def test_no_capabilities_when_the_residency_has_no_configured_provider():
    observability.setup_observability(_configured_settings())
    assert observability.instrumentation_capabilities("atlantis", True) == []


def test_no_capabilities_at_all_when_tracing_isnt_configured():
    assert observability.instrumentation_capabilities("eu", True) == []


def test_content_opt_in_flows_straight_into_instrumentation_settings():
    observability.setup_observability(_configured_settings())
    off = observability.instrumentation_capabilities("eu", False)
    on = observability.instrumentation_capabilities("eu", True)
    assert off[0].settings.include_content is False
    assert on[0].settings.include_content is True


def test_each_call_builds_a_fresh_instrumentation_never_a_shared_instance():
    observability.setup_observability(_configured_settings())
    first = observability.instrumentation_capabilities("eu", False)
    second = observability.instrumentation_capabilities("eu", True)
    assert first[0] is not second[0]
    assert first[0].settings is not second[0].settings


# --- resolve_tenant_tracing: a function of the tenant record (#105) ----------------------------


def _record(residency: str | None, opt_in: bool) -> TenantRecord:
    return TenantRecord(
        tenant_id=uuid.uuid4(),
        residency=residency,
        settings=TenantSettings(content_tracing_opt_in=opt_in),
    )


def test_resolve_tenant_tracing_reads_residency_and_content_opt_in_from_the_record():
    selection = observability.resolve_tenant_tracing(_record("eu", True))

    assert selection.residency == "eu"
    assert selection.content_tracing_opt_in is True


def test_resolve_tenant_tracing_defaults_to_untraced_content():
    selection = observability.resolve_tenant_tracing(_record("eu", False))

    assert selection.content_tracing_opt_in is False


def test_a_context_without_a_record_is_untraced():
    """A job or test context carries no record: nothing to trace it under, and no read here to
    find out (#105) -- untraced, content off."""
    selection = observability.resolve_tenant_tracing(None)

    assert selection == observability.TenantTracingSelection(
        residency=None, content_tracing_opt_in=False
    )


@pytest.mark.parametrize("residency", [None, "atlantis"])
def test_a_record_with_an_unresolved_residency_yields_no_capabilities(residency):
    """The one deliberate fail-open (ADR-0008): tracing on, content opted in, but no residency
    (or one with no sink) on the record -- the run goes untraced rather than failing or reaching
    another residency's sink."""
    observability.setup_observability(_configured_settings())
    selection = observability.resolve_tenant_tracing(_record(residency, True))

    assert (
        observability.instrumentation_capabilities(
            selection.residency, selection.content_tracing_opt_in
        )
        == []
    )


# --- tenant_span_attributes: only spans started inside the block are stamped -------------------


def test_tenant_span_attributes_only_stamps_spans_started_inside_the_block():
    exporter = InMemorySpanExporter()
    provider = observability.build_tracer_provider(exporter, processor_cls=SimpleSpanProcessor)
    tracer = provider.get_tracer("test")

    with observability.tenant_span_attributes({"tenant_id": "t-1", "identity_id": "u-1"}):
        with tracer.start_as_current_span("outer"):
            with tracer.start_as_current_span("inner"):
                pass
    with tracer.start_as_current_span("outside-the-block"):
        pass

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans["outer"].attributes["tenant_id"] == "t-1"
    assert spans["inner"].attributes["tenant_id"] == "t-1"
    assert "tenant_id" not in spans["outside-the-block"].attributes
