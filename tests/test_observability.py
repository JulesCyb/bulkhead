"""Unit tests for `app/observability.py` (Spec 8 / #62, ADR-0008): content-free-by-default
tracing, per-tenant opt-in, per-residency trace sinks, and a fail-loud startup guard — exercised
directly against the module (real `Settings`, fake control-plane sessions), no real database and
no real network call. The full ASGI-level, end-to-end behavior (spans an actual agent run
produces) lives in `tests/test_content_tracing.py`.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app import observability
from app.config import Settings
from app.context import RequestContext
from app.residency import ResidencyAllowList


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


# --- resolve_tenant_tracing_selection / resolve_tenant_tracing ----------------------------------


def _fake_session(residency_row, settings_row) -> AsyncMock:
    """Mirrors `tests/test_residency.py`'s `_fake_session`: `.execute(...).first()` returns
    whichever row is next, in the exact order `resolve_tenant_tracing_selection` issues its two
    queries (residency, then `tenants.settings`)."""
    rows = iter([residency_row, settings_row])

    async def _execute(*_args, **_kwargs):
        result = MagicMock()
        result.first.return_value = next(rows)
        return result

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=_execute)
    return session


async def test_resolve_tenant_tracing_selection_reads_residency_and_content_opt_in():
    ctx = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())
    session = _fake_session(("eu",), ({"content_tracing_opt_in": True},))

    selection = await observability.resolve_tenant_tracing_selection(session, ctx)

    assert selection.residency == "eu"
    assert selection.content_tracing_opt_in is True


async def test_resolve_tenant_tracing_selection_defaults_to_untraced_content():
    ctx = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())
    session = _fake_session((None,), (None,))

    selection = await observability.resolve_tenant_tracing_selection(session, ctx)

    assert selection.residency is None
    assert selection.content_tracing_opt_in is False


async def test_resolve_tenant_tracing_opens_no_session_when_tracing_is_off(monkeypatch):
    """`resolve_tenant_tracing` (the convenience entry point routes call) must not pay for a
    database round trip on every request in the overwhelmingly common case of tracing being off
    entirely."""

    def _fail_if_called(_ctx):
        raise AssertionError("no tenant session should be opened when tracing isn't configured")

    monkeypatch.setattr(observability, "tenant_session", _fail_if_called)
    ctx = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())

    selection = await observability.resolve_tenant_tracing(ctx)

    assert selection.residency is None
    assert selection.content_tracing_opt_in is False


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
