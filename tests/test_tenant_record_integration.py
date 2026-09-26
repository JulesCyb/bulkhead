"""The tenant record is read once per request (#104, spec #92 "A2"), against embedded Postgres.

Drives `app.main.app` over ASGI the way a client does, with the real, repository-backed
`ControlPlaneReads` adapter (never the suite-wide fake) and a SQLAlchemy `before_cursor_execute`
listener on the pooled engine that records every statement the request sends. What is asserted is
what the database sees: how many statements name the control-plane view `control.tenants_view`,
and whether any statement reaches a tenant table at all.

The search tool is replaced by one that opens a real `tenant_session(ctx)` and runs the real,
RLS-filtered `DocumentRepository.search` -- only the query embedding (a residency-routed network
call, `app.embeddings`) is skipped. That keeps a tenant-bound session inside the counted request,
so "the session layer consumes the record instead of reading the view again" is observed, not
assumed.

Model and tracing resolution are NOT stubbed (#105): the real `resolve_chat_model` resolves the
seeded tenant's model from its record -- residency, the deployment default model validated
against that residency's allow-list, and the gateway credential file the record's alias names
(written to a temporary directory) -- and the real `resolve_tenant_tracing` reads the same record.
Only the model *call* is replaced (`Agent.override(model=TestModel)`, which takes precedence over
the resolved model), since it would otherwise reach the gateway over the network. So the "exactly
one statement against the view" below counts the whole request, residency and credential alias
included: before #105 each of those was its own read of the view.
"""

from __future__ import annotations

import re
import time

import httpx
import jwt
import pytest
from pydantic_ai.models.test import TestModel
from sqlalchemy import event, text

pgserver = pytest.importorskip("pgserver")

from app import config  # noqa: E402
from app.agents import assistant as assistant_module  # noqa: E402
from app.agents.assistant import chat_assistant, one_shot_assistant  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402
from app.context import RequestContext  # noqa: E402
from app.context_resolution import FORBIDDEN_DETAIL  # noqa: E402
from app.db.session import get_engine, tenant_session  # noqa: E402
from app.main import app  # noqa: E402
from app.operator.suspend import set_tenant_suspended  # noqa: E402
from app.repositories.documents import DocumentHit, DocumentRepository  # noqa: E402
from app.token_verifier import set_default_adapter_for_tests  # noqa: E402
from tests.support import SeededTenant, cluster, environment, seed_tenant  # noqa: E402

_ = (cluster, environment)

_VIEW = re.compile(r"\bcontrol\.tenants_view\b")
_TENANT_TABLES = re.compile(r"\b(documents|memberships|conversations|messages)\b")

SECRET = "tenant-record-integration-secret-at-least-32-bytes"


@pytest.fixture(autouse=True)
def _real_control_plane_reads():
    """The real adapter for every test here -- the suite-wide fake would answer the record read
    from memory and nothing would reach the database to be counted."""
    set_default_adapter_for_tests(None)
    yield


@pytest.fixture
def statements(environment) -> list[str]:
    """Every statement the pooled engine sends, in order, for the duration of one test."""
    seen: list[str] = []

    def _record(conn, cursor, statement, parameters, context, executemany) -> None:
        seen.append(statement)

    engine = get_engine().sync_engine
    event.listen(engine, "before_cursor_execute", _record)
    yield seen
    event.remove(engine, "before_cursor_execute", _record)


_CREDENTIAL_ALIAS = "record-integration-gateway-key"


@pytest.fixture
def resolved_models(environment, monkeypatch, tmp_path) -> list[str]:
    """The real model resolution, end to end, minus the network: a gateway credential file under
    a temporary `GATEWAY_CREDENTIALS_DIR`, the real `resolve_chat_model` (wrapped only to record
    the bare model name it resolved), and both agents overridden with a `TestModel` for the call
    itself. A tenant must have `_CREDENTIAL_ALIAS` recorded (`_record_gateway_alias`)."""
    (tmp_path / _CREDENTIAL_ALIAS).write_text("sk-record-integration")
    monkeypatch.setenv("GATEWAY_CREDENTIALS_DIR", str(tmp_path))
    config.get_settings.cache_clear()

    seen: list[str] = []
    real_resolve = assistant_module.resolve_chat_model

    async def _recording_resolve(deps):
        model = await real_resolve(deps)
        seen.append(model.model_name)
        return model

    monkeypatch.setattr(assistant_module, "resolve_chat_model", _recording_resolve)
    model_call = TestModel(call_tools=["search_documents"])
    with one_shot_assistant.override(model=model_call), chat_assistant.override(model=model_call):
        yield seen
    config.get_settings.cache_clear()


async def _record_gateway_alias(tenant: SeededTenant) -> None:
    async with tenant.superuser_connection() as conn:
        async with conn.begin():
            await conn.execute(
                text(
                    "UPDATE control.tenants SET gateway_credential_alias = :a WHERE tenant_id = :t"
                ),
                {"a": _CREDENTIAL_ALIAS, "t": tenant.tenant_id},
            )


@pytest.fixture
def searched(monkeypatch, resolved_models) -> list[tuple[RequestContext, list[DocumentHit]]]:
    """Every (context, hits) the search tool produced -- through a real tenant-bound session."""
    seen: list[tuple[RequestContext, list[DocumentHit]]] = []

    async def _search(ctx: RequestContext, query: str, limit: int = 5) -> list[DocumentHit]:
        embedding = [0.0] * 1536
        embedding[0] = 1.0
        async with tenant_session(ctx) as session:
            hits = await DocumentRepository().search(session, embedding, limit=limit)
        seen.append((ctx, hits))
        return hits

    monkeypatch.setattr(assistant_module.document_tools, "search_documents", _search)
    return seen


async def _run(tenant: SeededTenant, headers: dict[str, str]) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            f"/v1/t/{tenant.tenant_id}/agents/assistant/run",
            json={"prompt": "Find the contract"},
            headers=headers,
        )


def _member_headers(tenant: SeededTenant) -> dict[str, str]:
    return {"X-Identity-Id": str(tenant.identities["member"]), "X-Roles": "member"}


def _view_reads(statements: list[str]) -> int:
    return sum(1 for s in statements if _VIEW.search(s))


async def _suspend(tenant: SeededTenant) -> None:
    """The operator's own `suspend` command body, as the owner role (ADR-0010)."""
    async with tenant.owner_connection() as conn:
        async with conn.begin():
            result = await set_tenant_suspended(conn, str(tenant.tenant_id), suspended=True)
    assert result.changed


async def test_one_request_reads_the_control_plane_view_exactly_once(
    environment, statements, searched, resolved_models
):
    """AC: one request issues exactly one statement against the control-plane view, and a second
    request issues one again -- no cross-request caching. Model resolution runs for real on each
    request (#105): its residency, model, and gateway credential alias come from that one read."""
    tenant = await seed_tenant(environment, residency="eu", roles=["member"], documents=1)
    await _record_gateway_alias(tenant)

    for request_number in (1, 2):
        statements.clear()
        response = await _run(tenant, _member_headers(tenant))

        assert response.status_code == 200, response.text
        assert _view_reads(statements) == 1, (request_number, statements)

    # The tool really did open a tenant-bound session inside each counted request, and that
    # session saw the tenant's own document -- routed from the record the context carried.
    assert [[hit.id for hit in hits] for _, hits in searched] == [tenant.document_ids] * 2
    record = searched[0][0].tenant_record
    assert record is not None
    assert (record.tenant_id, record.isolation_tier, record.residency) == (
        tenant.tenant_id,
        "pooled",
        "eu",
    )
    assert record.suspended_at is None
    assert record.gateway_credential_alias == _CREDENTIAL_ALIAS
    assert resolved_models == ["claude-eu", "claude-eu"]  # the deployment default, per request


async def test_suspending_between_two_requests_refuses_the_second_before_any_tenant_table(
    environment, statements, searched
):
    """AC: suspended between two requests, the second is refused with the generic 403, and not
    one statement reaches a tenant table for it."""
    tenant = await seed_tenant(environment, residency="eu", roles=["member"], documents=1)

    await _record_gateway_alias(tenant)
    first = await _run(tenant, _member_headers(tenant))
    assert first.status_code == 200, first.text
    await _suspend(tenant)

    statements.clear()
    second = await _run(tenant, _member_headers(tenant))

    assert second.status_code == 403
    assert second.json() == {"detail": FORBIDDEN_DETAIL}
    assert [s for s in statements if _TENANT_TABLES.search(s)] == []
    assert _view_reads(statements) == 1
    assert len(searched) == 1  # only the first request ever reached the tool


# --- AUTH_MODE=jwt: the production path, the record read before the membership lookup ---


@pytest.fixture
def jwt_mode():
    settings = Settings(
        _env_file=None,
        environment="test",
        auth_mode="jwt",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        default_identity_issuer="seed",
        jwt_verification_key=SECRET,
        jwt_algorithm="HS256",
    )
    app.dependency_overrides[get_settings] = lambda: settings
    yield settings
    app.dependency_overrides.pop(get_settings, None)


def _bearer(tenant: SeededTenant) -> dict[str, str]:
    """A person's token for the seeded member: `seed_tenant` records every identity under issuer
    `seed` with its own id as the subject."""
    now = int(time.time())
    claims = {
        "iss": "seed",
        "sub": str(tenant.identities["member"]),
        "aud": str(tenant.tenant_id),
        "iat": now,
        "exp": now + 300,
    }
    return {"Authorization": f"Bearer {jwt.encode(claims, SECRET, algorithm='HS256')}"}


async def test_a_bearer_request_reads_the_control_plane_view_exactly_once(
    environment, statements, searched, resolved_models, jwt_mode
):
    """AC on the production path: the record is read right after the token verifies (no database
    needed for that) and handed to the membership lookup, whose session routes from it -- so the
    membership lookup, the tool's session, and everything after share the one view read. A second
    request reads it once again."""
    tenant = await seed_tenant(environment, residency="eu", roles=["member"], documents=1)
    await _record_gateway_alias(tenant)

    for request_number in (1, 2):
        statements.clear()
        response = await _run(tenant, _bearer(tenant))

        assert response.status_code == 200, response.text
        assert _view_reads(statements) == 1, (request_number, statements)

    assert [[hit.id for hit in hits] for _, hits in searched] == [tenant.document_ids] * 2
    assert resolved_models == ["claude-eu", "claude-eu"]


async def test_a_suspended_tenants_bearer_request_is_refused_with_the_generic_403(
    environment, statements, searched, jwt_mode
):
    """Suspension under a bearer token: the record read right after the token verifies refuses
    it -- before the identity or membership lookup, before any tenant table -- with the generic
    403, never the unhandled-exception 500."""
    tenant = await seed_tenant(environment, residency="eu", roles=["member"], documents=1)
    await _suspend(tenant)

    statements.clear()
    response = await _run(tenant, _bearer(tenant))

    assert response.status_code == 403, response.text
    assert response.json() == {"detail": FORBIDDEN_DETAIL}
    assert [s for s in statements if _TENANT_TABLES.search(s)] == []
    assert not any("identity_lookup" in s for s in statements)
    assert _view_reads(statements) == 1
    assert searched == []


async def test_a_token_for_another_tenant_never_reaches_the_database(
    environment, statements, searched, jwt_mode
):
    """The record read comes after the audience check: a token whose audience is not the path's
    tenant is refused without a single statement against the view (an unauthenticated caller
    still cannot make the server read the control plane)."""
    tenant = await seed_tenant(environment, residency="eu", roles=["member"], documents=1)
    other = await seed_tenant(environment, residency="eu", roles=["member"])

    statements.clear()
    response = await _run(tenant, _bearer(other))

    assert response.status_code == 403, response.text
    assert _view_reads(statements) == 0, statements
