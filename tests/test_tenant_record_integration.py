"""The tenant record is read once per request (#104, spec #92 "A2"), against embedded Postgres.

Drives `app.main.app` over ASGI the way a client does, with the real, repository-backed
`ControlPlaneReads` adapter (never the suite-wide fake) and a SQLAlchemy `before_cursor_execute`
listener on the pooled engine that records every statement the request sends. What is asserted is
what the database sees: how many statements name the control-plane view `control.tenants_view`,
and whether any statement reaches a tenant table at all.

The search tool is replaced by one that opens a real `tenant_session(ctx)` and runs the real,
RLS-filtered `DocumentRepository.search` -- only the query embedding (a residency-routed network
call, `app.embeddings`, rewritten onto the record by #105) is skipped. That keeps a tenant-bound
session inside the counted request, so "the session layer consumes the record instead of reading
the view again" is observed, not assumed. The model resolver is replaced by `TestModel` through
the shared `test_model` fixture for the same reason (#105); tracing is unconfigured in the suite,
so `resolve_tenant_tracing` opens no session at all.
"""

from __future__ import annotations

import re
import time

import httpx
import jwt
import pytest
from sqlalchemy import event

pgserver = pytest.importorskip("pgserver")

from app.agents import assistant as assistant_module  # noqa: E402
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


@pytest.fixture
def searched(monkeypatch, test_model) -> list[tuple[RequestContext, list[DocumentHit]]]:
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
    environment, statements, searched
):
    """AC: one request issues exactly one statement against the control-plane view, and a second
    request issues one again -- no cross-request caching."""
    tenant = await seed_tenant(environment, residency="eu", roles=["member"], documents=1)

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


async def test_suspending_between_two_requests_refuses_the_second_before_any_tenant_table(
    environment, statements, searched
):
    """AC: suspended between two requests, the second is refused with the generic 403, and not
    one statement reaches a tenant table for it."""
    tenant = await seed_tenant(environment, residency="eu", roles=["member"], documents=1)

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


# --- AUTH_MODE=jwt: the production path, with the membership read before the record ---


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


async def test_a_bearer_request_reads_the_view_once_for_the_membership_and_once_for_the_record(
    environment, statements, searched, jwt_mode
):
    """The bearer chain looks the membership up (spec #92: the record is read *after* it) through
    a role-free, record-less context whose tenant session routes itself -- one view read -- and
    then reads the record -- the second. Every session after that consumes the record. This test
    pins today's exact count, 2, so a regression to 3 is caught; bringing it to 1 means reading
    the record before the membership lookup and handing it to that lookup (reported on #104)."""
    tenant = await seed_tenant(environment, residency="eu", roles=["member"], documents=1)

    for _request_number in (1, 2):
        statements.clear()
        response = await _run(tenant, _bearer(tenant))

        assert response.status_code == 200, response.text
        assert _view_reads(statements) == 2, statements

    assert [[hit.id for hit in hits] for _, hits in searched] == [tenant.document_ids] * 2


async def test_a_suspended_tenants_bearer_request_is_refused_with_the_generic_403(
    environment, statements, searched, jwt_mode
):
    """Suspension under a bearer token: the membership lookup's own routing read already sees the
    suspension, before any tenant table is touched -- and the answer is the generic 403, never
    the unhandled-exception 500."""
    tenant = await seed_tenant(environment, residency="eu", roles=["member"], documents=1)
    await _suspend(tenant)

    statements.clear()
    response = await _run(tenant, _bearer(tenant))

    assert response.status_code == 403, response.text
    assert response.json() == {"detail": FORBIDDEN_DETAIL}
    assert [s for s in statements if _TENANT_TABLES.search(s)] == []
    assert searched == []
