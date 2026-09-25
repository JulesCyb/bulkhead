"""ASGI-seam tests for S3-T1 / #26: the role-gated membership-listing route, the worked example
for ADR-0004's "roles gate actions, never visibility."

`app/tools/memberships.py` and the tenant repository it calls are faked here (no real Postgres),
the same pattern `tests/test_jwt_auth.py` uses for the control plane -- the real membership table,
its role CHECK constraint, and RLS have their own coverage in `tests/test_rls_integration.py`.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import httpx
import jwt
import pytest

import app.tenant_suspension as tenant_suspension_module
import app.token_verifier as token_verifier_module
import app.tools.memberships as memberships_tools_module
from app.config import Settings, get_settings
from app.context import RequestContext
from app.main import app
from app.repositories.memberships import MembershipRecord

SECRET = "asgi-jwt-test-shared-secret-at-least-32-bytes"
ISSUER = "https://idp.example.com"


def _record(*, identity_id: uuid.UUID, role: str) -> MembershipRecord:
    return MembershipRecord(
        id=uuid.uuid4(),
        identity_id=identity_id,
        role=role,
        created_at=datetime.now(UTC),
    )


@asynccontextmanager
async def _fake_session():
    yield None


def _install_fake_memberships(monkeypatch, records_by_tenant: dict[uuid.UUID, list]):
    class FakeMembershipRepository:
        async def list_for_tenant(self, session, ctx: RequestContext):
            return records_by_tenant.get(ctx.tenant_id, [])

    monkeypatch.setattr(memberships_tools_module, "tenant_session", lambda ctx: _fake_session())
    monkeypatch.setattr(memberships_tools_module, "MembershipRepository", FakeMembershipRepository)


def _headers(identity_id: uuid.UUID, roles: str) -> dict[str, str]:
    return {"X-Identity-Id": str(identity_id), "X-Roles": roles}


async def _get_memberships(tenant_id: uuid.UUID, headers: dict[str, str]) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(f"/v1/t/{tenant_id}/memberships", headers=headers)


# --- Admin gets the list; every other role is refused with 403 naming 'admin' ---


async def test_admin_lists_the_tenants_memberships(monkeypatch):
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    other_identity = uuid.uuid4()
    records = [
        _record(identity_id=identity_id, role="admin"),
        _record(identity_id=other_identity, role="member"),
    ]
    _install_fake_memberships(monkeypatch, {tenant_id: records})

    response = await _get_memberships(tenant_id, _headers(identity_id, "admin"))

    assert response.status_code == 200, response.text
    body = response.json()
    returned = body["memberships"]
    assert len(returned) == 2
    assert {r["identity_id"] for r in returned} == {str(identity_id), str(other_identity)}
    assert {r["role"] for r in returned} == {"admin", "member"}
    assert all("created_at" in r for r in returned)


@pytest.mark.parametrize("role", ["member", "support", "agent"])
async def test_non_admin_role_is_refused_with_403_naming_admin(monkeypatch, role):
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    _install_fake_memberships(monkeypatch, {tenant_id: []})

    response = await _get_memberships(tenant_id, _headers(identity_id, role))

    assert response.status_code == 403
    body = response.json()
    assert body["error"] == "forbidden"
    assert "admin" in body["message"]


# --- The role-check failure is served by the registered exception handler, not an unhandled 500 ---


async def test_denied_role_check_is_handled_not_unhandled(monkeypatch):
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    _install_fake_memberships(monkeypatch, {tenant_id: []})

    # raise_app_exceptions=False would only matter if the exception propagated -- asserting a
    # clean 403 (not a raised exception reaching this test) is itself the regression proof.
    response = await _get_memberships(tenant_id, _headers(identity_id, "member"))

    assert response.status_code == 403
    assert response.json()["error"] == "forbidden"


# --- The denial is logged with identifiers only, never request content ---


async def test_denied_role_check_is_logged_with_identifiers_only(monkeypatch, caplog):
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    _install_fake_memberships(monkeypatch, {tenant_id: []})

    with caplog.at_level(logging.WARNING):
        response = await _get_memberships(tenant_id, _headers(identity_id, "member"))

    assert response.status_code == 403
    matching = [r for r in caplog.records if getattr(r, "event", None) == "role_check_denied"]
    assert matching, "expected a role_check_denied log record"
    record = matching[0]
    assert record.tenant_id == str(tenant_id)
    assert record.identity_id == str(identity_id)
    assert record.required_role == "admin"
    # No request content (headers, body, URLs) leaks into the log line's own text.
    assert str(identity_id) not in record.getMessage()
    assert "memberships" not in record.getMessage().lower()


# --- A role change in the database takes effect on the very next request ---


async def test_role_change_takes_effect_on_the_next_request_no_refresh(monkeypatch):
    """dev-headers mode builds a brand-new RequestContext per request straight from the current
    header -- there is no session or cache in between, so the very next request with a new role
    header is, by construction, served with the new role. The equivalent property in jwt mode
    (no caching of the membership row) is covered in test_jwt_auth.py / below."""
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    _install_fake_memberships(monkeypatch, {tenant_id: []})

    first = await _get_memberships(tenant_id, _headers(identity_id, "member"))
    assert first.status_code == 403

    second = await _get_memberships(tenant_id, _headers(identity_id, "admin"))
    assert second.status_code == 200


# --- JWT mode: role always comes from the real membership row, never a spoofed header ---


def _make_token(*, subject: str, audience: str, exp_delta: float = 300.0) -> str:
    now = int(time.time())
    claims = {
        "iss": ISSUER,
        "sub": subject,
        "aud": audience,
        "iat": now,
        "exp": now + exp_delta,
    }
    return jwt.encode(claims, SECRET, algorithm="HS256")


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
    )


def _install_fake_control_plane(monkeypatch, *, auth_settings, identities, memberships):
    from types import SimpleNamespace

    class FakeTenantAuthSettingsRepository:
        async def get(self, session, *, tenant_id, default_issuer=None):
            if tenant_id not in auth_settings:
                return None
            issuer, suspended = auth_settings[tenant_id]
            return SimpleNamespace(issuer=issuer or default_issuer, suspended=suspended)

    class FakeIdentityRepository:
        async def find_by_issuer_and_subject(self, session, *, issuer, subject):
            identity_id = identities.get((issuer, subject))
            if identity_id is None:
                return None
            return SimpleNamespace(id=identity_id, issuer=issuer, subject=subject)

    class FakeMembershipRepository:
        async def get_role(self, session, ctx: RequestContext, *, identity_id):
            return memberships.get((ctx.tenant_id, identity_id))

    # The token check lives in app.token_verifier (#44); the suspension check app.deps calls lives
    # in app.tenant_suspension (#69) -- app.deps keeps neither copy itself.
    for module in (tenant_suspension_module, token_verifier_module):
        monkeypatch.setattr(module, "control_session", _fake_session)
        monkeypatch.setattr(
            module, "TenantAuthSettingsRepository", FakeTenantAuthSettingsRepository
        )
    monkeypatch.setattr(token_verifier_module, "tenant_session", lambda ctx: _fake_session())
    monkeypatch.setattr(token_verifier_module, "IdentityRepository", FakeIdentityRepository)
    monkeypatch.setattr(token_verifier_module, "MembershipRepository", FakeMembershipRepository)


@pytest.fixture
def jwt_client(monkeypatch):
    settings = _jwt_settings()
    app.dependency_overrides[get_settings] = lambda: settings
    yield settings
    app.dependency_overrides.pop(get_settings, None)


async def test_jwt_mode_spoofed_role_header_is_ignored_member_still_refused(
    jwt_client, monkeypatch
):
    """A verified token for role `member`, plus a spoofed X-Roles: admin header alongside it,
    must still resolve to `member` from the real membership row and be refused on the admin
    route -- the header is never consulted once a real token is in effect."""
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    _install_fake_control_plane(
        monkeypatch,
        auth_settings={tenant_id: (ISSUER, False)},
        identities={(ISSUER, "sub-1"): identity_id},
        memberships={(tenant_id, identity_id): "member"},
    )
    _install_fake_memberships(monkeypatch, {tenant_id: []})

    token = _make_token(subject="sub-1", audience=str(tenant_id))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            f"/v1/t/{tenant_id}/memberships",
            headers={"Authorization": f"Bearer {token}", "X-Roles": "admin"},
        )

    assert response.status_code == 403
    assert response.json()["error"] == "forbidden"


async def test_jwt_mode_role_promotion_takes_effect_next_request_no_token_refresh(
    jwt_client, monkeypatch
):
    """Same token used twice; the underlying membership row is promoted from `member` to `admin`
    in between -- no new token, no session. The second request must reflect the new role."""
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    memberships = {(tenant_id, identity_id): "member"}
    _install_fake_control_plane(
        monkeypatch,
        auth_settings={tenant_id: (ISSUER, False)},
        identities={(ISSUER, "sub-1"): identity_id},
        memberships=memberships,
    )
    _install_fake_memberships(monkeypatch, {tenant_id: []})
    token = _make_token(subject="sub-1", audience=str(tenant_id))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.get(
            f"/v1/t/{tenant_id}/memberships", headers={"Authorization": f"Bearer {token}"}
        )
        assert first.status_code == 403

        memberships[(tenant_id, identity_id)] = "admin"

        second = await client.get(
            f"/v1/t/{tenant_id}/memberships", headers={"Authorization": f"Bearer {token}"}
        )
        assert second.status_code == 200, second.text
