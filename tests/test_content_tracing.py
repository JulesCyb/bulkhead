"""ASGI-level tests for content-free-by-default tracing, per-tenant content opt-in, and
per-residency trace sinks (Spec 8 / #62, ADR-0008) — a real running app (`app.main.app`), a
`TestModel` that actually calls the search tool, and a fake in-memory span exporter swapped in
per residency (`app.observability.set_tracer_provider_for_residency`).
Each tenant's residency and content opt-in come from its tenant record (#105), installed through
the shared control-plane fake (`tests.conftest.FakeControlPlaneReads.records`) that context
resolution reads it from -- no database, no real model call anywhere.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai.models.test import TestModel

from app import observability
from app.main import app
from app.repositories.documents import DocumentHit
from app.tenant_record import TenantRecord
from app.tenant_settings import TenantSettings
from app.token_verifier import set_default_adapter_for_tests
from tests.conftest import FakeControlPlaneReads

PROMPT = "What does the contract say?"
DOCUMENT_TITLE = "Acme confidential contract"
DOCUMENT_SNIPPET = "the secret indemnification clause"


@pytest.fixture(autouse=True)
def _reset_tracer_providers():
    observability.reset_tracer_providers()
    yield
    observability.reset_tracer_providers()


@pytest.fixture
def content_search():
    """A fake search tool whose result carries content a test can grep captured spans for —
    exactly the kind of document content ADR-0008 says must never appear unless the tenant has
    opted in."""

    async def _search(ctx, query, limit):
        return [
            DocumentHit(id=uuid.uuid4(), title=DOCUMENT_TITLE, snippet=DOCUMENT_SNIPPET, score=0.9)
        ]

    return _search


@pytest.fixture
def client(route_run, content_search):
    route_run(TestModel(call_tools=["search_documents"]), search=content_search)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _install_records(directory: dict[uuid.UUID, tuple[str | None, bool]]) -> None:
    """Each tenant's `(residency, content_tracing_opt_in)` as its tenant record -- what context
    resolution attaches to the request and the routes take tracing from (#105)."""
    set_default_adapter_for_tests(
        FakeControlPlaneReads(
            records={
                tenant_id: TenantRecord(
                    tenant_id=tenant_id,
                    residency=residency,
                    settings=TenantSettings(content_tracing_opt_in=opt_in),
                )
                for tenant_id, (residency, opt_in) in directory.items()
            }
        )
    )


def _all_span_text(spans) -> str:
    """Every attribute value of every captured span, concatenated — the corpus a "must never
    appear" / "must appear" assertion greps."""
    return "\n".join(str(value) for span in spans for value in span.attributes.values())


async def test_untraced_by_default_produces_no_prompt_or_tool_result_content(client, monkeypatch):
    """AC1: an agent run for a tenant that has not opted in produces zero prompt or tool-result
    text across every span the run produces."""
    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    exporter = InMemorySpanExporter()
    provider = observability.build_tracer_provider(exporter, processor_cls=SimpleSpanProcessor)
    observability.set_tracer_provider_for_residency("eu", provider)
    _install_records({tenant_id: ("eu", False)})

    async with client:
        response = await client.post(
            f"/v1/t/{tenant_id}/agents/assistant/run",
            json={"prompt": PROMPT},
            headers={"X-Identity-Id": str(identity_id)},
        )
    assert response.status_code == 200, response.text

    spans = exporter.get_finished_spans()
    assert spans, "the run must produce at least one span"
    text = _all_span_text(spans)
    assert PROMPT not in text
    assert DOCUMENT_TITLE not in text
    assert DOCUMENT_SNIPPET not in text


async def test_content_opt_in_tenant_produces_that_content_in_the_spans(client, monkeypatch):
    """AC2: the same run for a tenant with the content opt-in set produces that content in the
    captured spans."""
    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    exporter = InMemorySpanExporter()
    provider = observability.build_tracer_provider(exporter, processor_cls=SimpleSpanProcessor)
    observability.set_tracer_provider_for_residency("eu", provider)
    _install_records({tenant_id: ("eu", True)})

    async with client:
        response = await client.post(
            f"/v1/t/{tenant_id}/agents/assistant/run",
            json={"prompt": PROMPT},
            headers={"X-Identity-Id": str(identity_id)},
        )
    assert response.status_code == 200, response.text

    text = _all_span_text(exporter.get_finished_spans())
    assert PROMPT in text
    assert DOCUMENT_SNIPPET in text


async def test_every_span_carries_tenant_and_user_identifiers(client, monkeypatch):
    """AC3: every span the run produces carries tenant and user identifiers as span attributes,
    not only as metadata on the root span."""
    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    exporter = InMemorySpanExporter()
    provider = observability.build_tracer_provider(exporter, processor_cls=SimpleSpanProcessor)
    observability.set_tracer_provider_for_residency("eu", provider)
    _install_records({tenant_id: ("eu", False)})

    async with client:
        response = await client.post(
            f"/v1/t/{tenant_id}/agents/assistant/run",
            json={"prompt": PROMPT},
            headers={"X-Identity-Id": str(identity_id)},
        )
    assert response.status_code == 200, response.text

    spans = exporter.get_finished_spans()
    # The run's tool call and model request each produce their own span, on top of the root
    # agent-run span -- proving the attributes aren't only on that root span.
    assert len(spans) > 1
    for span in spans:
        assert span.attributes["tenant_id"] == str(tenant_id)
        assert span.attributes["identity_id"] == str(identity_id)


async def test_two_tenants_different_residencies_use_two_distinct_sinks(client, monkeypatch):
    """AC4: runs for two tenants set to different residencies export their spans to two distinct
    sinks in the same test process, resolved through `app.observability.resolve_tenant_tracing`
    -- the same resolver-composition path (residency -> route) the model and embedding routes
    use (`app.residency.resolve_residency_route`) -- so "no tenant's spans cross residencies" is
    proven, not assumed.
    """
    eu_tenant, eu_identity = uuid.uuid4(), uuid.uuid4()
    us_tenant, us_identity = uuid.uuid4(), uuid.uuid4()

    eu_exporter = InMemorySpanExporter()
    us_exporter = InMemorySpanExporter()
    observability.set_tracer_provider_for_residency(
        "eu", observability.build_tracer_provider(eu_exporter, processor_cls=SimpleSpanProcessor)
    )
    observability.set_tracer_provider_for_residency(
        "us", observability.build_tracer_provider(us_exporter, processor_cls=SimpleSpanProcessor)
    )
    _install_records({eu_tenant: ("eu", False), us_tenant: ("us", False)})

    async with client:
        eu_response = await client.post(
            f"/v1/t/{eu_tenant}/agents/assistant/run",
            json={"prompt": PROMPT},
            headers={"X-Identity-Id": str(eu_identity)},
        )
        us_response = await client.post(
            f"/v1/t/{us_tenant}/agents/assistant/run",
            json={"prompt": PROMPT},
            headers={"X-Identity-Id": str(us_identity)},
        )
    assert eu_response.status_code == 200, eu_response.text
    assert us_response.status_code == 200, us_response.text

    eu_spans = eu_exporter.get_finished_spans()
    us_spans = us_exporter.get_finished_spans()
    assert eu_spans and us_spans

    eu_tenant_ids = {span.attributes["tenant_id"] for span in eu_spans}
    us_tenant_ids = {span.attributes["tenant_id"] for span in us_spans}
    assert eu_tenant_ids == {str(eu_tenant)}
    assert us_tenant_ids == {str(us_tenant)}


async def test_tenant_with_no_resolvable_residency_runs_untraced_not_crashed(client, monkeypatch):
    """A tenant whose residency can't be resolved must still be served normally -- tracing is a
    tracing-only concern and a run left untraced trivially can't cross a residency boundary or
    leak content."""
    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    exporter = InMemorySpanExporter()
    observability.set_tracer_provider_for_residency(
        "eu", observability.build_tracer_provider(exporter, processor_cls=SimpleSpanProcessor)
    )
    _install_records({tenant_id: (None, False)})

    async with client:
        response = await client.post(
            f"/v1/t/{tenant_id}/agents/assistant/run",
            json={"prompt": PROMPT},
            headers={"X-Identity-Id": str(identity_id)},
        )
    assert response.status_code == 200, response.text
    assert list(exporter.get_finished_spans()) == []
