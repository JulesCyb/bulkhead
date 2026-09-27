"""ASGI-seam tests for the one context-resolution module (#101, spec #91): the HTTP adapter
(`app.deps.get_context`) under both `AUTH_MODE` values delegates the whole chain -- bearer parsing,
verification, suspension, means -- to `app.context_resolution` and only renders what it returns.

Everything here drives `app.main.app` through `httpx.ASGITransport` the way a client does and
asserts on what a client, a tool, or a trace can observe (status, body, the tool's received
context, the span attributes, the security-event log line). The control plane is faked through the
one shared `FakeControlPlaneReads` adapter (`tests/conftest.py`), never a monkeypatched repository.
"""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path

import httpx
import jwt
import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai.models.test import TestModel

import app.context_resolution as context_resolution_module
import app.deps as deps_module
from app import observability
from app.config import Settings, get_settings
from app.context import Means
from app.context_resolution import ContextRejection, RejectionReason, RejectionStatus
from app.main import app
from app.tenant_record import TenantRecord
from app.token_verifier import (
    AGENT_IDENTITY_ISSUER,
    ResolvedIdentity,
    set_default_adapter_for_tests,
)
from tests.conftest import FakeControlPlaneReads

SECRET = "asgi-jwt-test-shared-secret-at-least-32-bytes"
AGENT_SECRET = "asgi-agent-token-test-secret-at-least-32-bytes"
ISSUER = "https://idp.example.com"


def _jwt_settings() -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        auth_mode="jwt",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        default_identity_issuer=None,
        jwt_verification_key=SECRET,
        jwt_algorithm="HS256",
        agent_token_signing_key=AGENT_SECRET,
    )


@pytest.fixture
def jwt_mode():
    settings = _jwt_settings()
    app.dependency_overrides[get_settings] = lambda: settings
    yield settings
    app.dependency_overrides.pop(get_settings, None)


def _token(
    *,
    audience: str,
    issuer: str = ISSUER,
    subject: str = "sub-1",
    secret: str = SECRET,
    extra: dict | None = None,
) -> str:
    now = int(time.time())
    claims = {"iss": issuer, "sub": subject, "aud": audience, "iat": now, "exp": now + 300}
    claims.update(extra or {})
    return jwt.encode(claims, secret, algorithm="HS256")


async def _post_run(tenant_id: uuid.UUID, headers: dict[str, str]) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            f"/v1/t/{tenant_id}/agents/assistant/run", json={"prompt": "Hi"}, headers=headers
        )


@pytest.fixture
def run_route(route_run):
    """The one-shot run route with a fake search tool that records the context it received."""
    route_run(TestModel(call_tools=["search_documents"]))


# --- Delegation proof: each adapter renders whatever the module returns, no second copy ---


def _substitute_resolution(monkeypatch, outcome) -> list[dict]:
    """Replaces the module's one entry point; returns the list of keyword arguments it was
    called with, so a test can also see what the adapter handed over."""
    seen: list[dict] = []

    async def _fake(**kwargs):
        seen.append(kwargs)
        return outcome

    monkeypatch.setattr(context_resolution_module, "resolve_request_context", _fake)
    return seen


async def test_dev_headers_adapter_renders_a_rejection_it_did_not_compute(monkeypatch, caplog):
    """The dev-headers twin of `tests/test_jwt_auth.py`'s delegation proof: a perfectly valid
    X-Identity-Id for an unsuspended tenant, yet the substituted module says the tenant is
    suspended -- and the adapter answers exactly that."""
    tenant_id = uuid.uuid4()
    rejection = ContextRejection(
        status=RejectionStatus.FORBIDDEN,
        reason=RejectionReason.TENANT_SUSPENDED,
        detail=deps_module.FORBIDDEN_DETAIL,
        request_id="req-dev",
        issuer="dev-headers",
    )
    _substitute_resolution(monkeypatch, rejection)

    with caplog.at_level("WARNING"):
        response = await _post_run(tenant_id, {"X-Identity-Id": str(uuid.uuid4())})

    assert response.status_code == 403
    assert response.json()["detail"] == deps_module.FORBIDDEN_DETAIL
    reasons = [
        r.reason for r in caplog.records if getattr(r, "event", None) == "jwt_auth_forbidden"
    ]
    assert reasons == ["tenant_suspended"]


async def test_dev_headers_adapter_hands_on_the_context_the_module_returned(
    monkeypatch, run_route, contexts
):
    """Success path of the twin: whatever context the module returns is what the tool receives
    -- here with a role and a means no dev header asked for."""
    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    substituted = context_resolution_module.actor_context(
        tenant_id,
        ResolvedIdentity(
            identity_id=identity_id, role="support", issuer="agent", credential_public_id="c-1"
        ),
        # The real module always attaches the tenant record it read (#104); a run needs it.
        tenant_record=TenantRecord.pooled_default(tenant_id),
    )
    _substitute_resolution(monkeypatch, substituted)

    response = await _post_run(tenant_id, {"X-Identity-Id": str(uuid.uuid4()), "X-Roles": "admin"})

    assert response.status_code == 200, response.text
    assert contexts and contexts[0] == substituted
    assert response.headers["X-Request-Id"] == substituted.request_id


# --- HTTP contexts carry a means for the first time ---


async def test_dev_headers_request_reaches_the_tool_as_delegation(run_route, contexts):
    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    response = await _post_run(tenant_id, {"X-Identity-Id": str(identity_id), "X-Roles": "member"})

    assert response.status_code == 200, response.text
    ctx = contexts[0]
    assert (ctx.tenant_id, ctx.identity_id, ctx.roles) == (tenant_id, identity_id, {"member"})
    assert ctx.means == Means(kind="agent", id="assistant")


async def test_a_persons_token_reaches_the_tool_as_delegation(jwt_mode, run_route, contexts):
    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    set_default_adapter_for_tests(
        FakeControlPlaneReads(
            auth_settings={tenant_id: (ISSUER, False)},
            identities={(ISSUER, "sub-1"): identity_id},
            memberships={(tenant_id, identity_id): "member"},
        )
    )
    response = await _post_run(
        tenant_id, {"Authorization": f"Bearer {_token(audience=str(tenant_id))}"}
    )

    assert response.status_code == 200, response.text
    assert contexts[0].identity_id == identity_id
    assert contexts[0].means == Means(kind="agent", id="assistant")


async def test_an_agent_identitys_token_reaches_the_tool_naming_its_credential(
    jwt_mode, run_route, contexts
):
    tenant_id, agent_identity_id = uuid.uuid4(), uuid.uuid4()
    set_default_adapter_for_tests(
        FakeControlPlaneReads(
            identities={(AGENT_IDENTITY_ISSUER, "agent-sub"): agent_identity_id},
            memberships={(tenant_id, agent_identity_id): "agent"},
        )
    )
    token = _token(
        audience=str(tenant_id),
        issuer=AGENT_IDENTITY_ISSUER,
        subject="agent-sub",
        secret=AGENT_SECRET,
        extra={"cred": "agt_public_1"},
    )
    response = await _post_run(tenant_id, {"Authorization": f"Bearer {token}"})

    assert response.status_code == 200, response.text
    assert contexts[0].roles == frozenset({"agent"})
    assert contexts[0].means == Means(kind="credential", id="agt_public_1")


async def test_every_span_of_an_http_run_records_the_person_as_actor_and_the_agent_as_means(
    monkeypatch, run_route
):
    """The audit fact an operator reads for an HTTP-originated run (CONTEXT.md "Delegation":
    the person as the actor, the agent as the means) -- on every span, not only the root."""
    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    observability.reset_tracer_providers()
    exporter = InMemorySpanExporter()
    provider = observability.build_tracer_provider(exporter, processor_cls=SimpleSpanProcessor)
    observability.set_tracer_provider_for_residency("eu", provider)
    # Residency "eu", no content opt-in, on the tenant record context resolution reads (#105).
    set_default_adapter_for_tests(
        FakeControlPlaneReads(
            records={tenant_id: TenantRecord(tenant_id=tenant_id, residency="eu")}
        )
    )
    try:
        response = await _post_run(tenant_id, {"X-Identity-Id": str(identity_id)})
    finally:
        observability.reset_tracer_providers()

    assert response.status_code == 200, response.text
    spans = exporter.get_finished_spans()
    assert len(spans) > 1
    for span in spans:
        assert span.attributes["identity_id"] == str(identity_id)
        assert span.attributes["means_kind"] == "agent"
        assert span.attributes["means_id"] == "assistant"


# --- Only the module builds a RequestContext ---

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every place outside tests/ allowed to call `RequestContext(...)`, with how many times and why.
# A new construction site anywhere -- above all in an adapter (app/deps.py, app/api/, the MCP
# middleware) -- fails this test until it is either routed through `app.context_resolution` or
# added here with a reason a reviewer accepts.
ALLOWED_CONSTRUCTION_SITES: dict[str, int] = {
    # The one module that turns a request's inputs into the context it acts as: `_new_context`
    # (the bearer/dev-headers chain) plus `resolve_stdio_env_context` (#102 -- the stdio
    # transport's own development-only environment identity, moved here from app/mcp/server.py).
    "app/context_resolution.py": 2,
    # Role-free, never-returned context that only opens the tenant session for the membership
    # read inside the resolution chain (issue #91 story 23) -- internal to the verifier adapter.
    "app/token_verifier.py": 1,
    # Same shape for the agent-credential exchange: no identity is known yet, nothing returned.
    "app/agent_credential_exchange.py": 1,
    # A job, not a request: the retention job's own fixed job identity (ADR-0006, rule 1).
    "app/retention.py": 1,
    # #102 moved the stdio-only development fallback into `resolve_stdio_env_context` above;
    # `app/mcp/server.py`'s `_context_from_env` is now a thin wrapper with no construction site
    # of its own, so it no longer appears here at all.
}


def test_request_context_is_constructed_only_at_the_allowed_sites():
    pattern = re.compile(r"\bRequestContext\(")
    found: dict[str, int] = {}
    for directory in ("app", "scripts", "migrations"):
        for path in (REPO_ROOT / directory).rglob("*.py"):
            code_lines = [
                line for line in path.read_text().splitlines() if not line.lstrip().startswith("#")
            ]
            count = sum(len(pattern.findall(line)) for line in code_lines)
            if count:
                found[path.relative_to(REPO_ROOT).as_posix()] = count
    assert found == ALLOWED_CONSTRUCTION_SITES


# --- Bearer parsing exists exactly once (#102 acceptance criterion) ---


def test_bearer_parsing_exists_only_in_the_context_resolution_module():
    """`parse_bearer_token` (module docstring) is the only place a raw `Authorization` header is
    turned into a token -- the MCP middleware (`app/mcp/server.py`) used to keep its own copy of
    exactly this parsing before #102; a second copy anywhere is exactly the kind of parsing
    difference between transports a security reviewer would call a bypass (issue #91 story 21)."""
    needles = ('startswith("bearer ")', 'split(" ", 1)')
    found: set[str] = set()
    for path in (REPO_ROOT / "app").rglob("*.py"):
        text = path.read_text()
        if any(needle in text for needle in needles):
            found.add(path.relative_to(REPO_ROOT).as_posix())
    assert found == {"app/context_resolution.py"}
