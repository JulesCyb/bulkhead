"""Unit tests for the shared tenant-token verifier (#44): exercises `verify_tenant_token`
directly, with the control-plane and membership repositories faked (no real Postgres) -- the same
faking pattern `tests/test_jwt_auth.py` uses at the ASGI seam, but here calling the module
directly so each categorized failure is proven in isolation from FastAPI/HTTP entirely.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import jwt
import pytest

import app.token_verifier as token_verifier_module
from app.context import RequestContext
from app.jwt_verifier import mint_token
from app.token_verifier import (
    AGENT_IDENTITY_ISSUER,
    TenantTokenVerificationError,
    VerificationFailureReason,
    verify_tenant_token,
)

SECRET = "token-verifier-unit-test-shared-secret-32-bytes"
ISSUER = "https://idp.example.com"


def _key_source(issuer: str, kid: str | None) -> str:
    return SECRET


def _make_token(
    *,
    secret: str = SECRET,
    issuer: str = ISSUER,
    subject: str = "sub-1",
    audience: str | None = None,
    exp_delta: float = 300.0,
    algorithm: str = "HS256",
) -> str:
    now = int(time.time())
    claims = {
        "iss": issuer,
        "sub": subject,
        "aud": audience,
        "iat": now,
        "exp": now + exp_delta,
    }
    return jwt.encode(claims, secret, algorithm=algorithm)


@asynccontextmanager
async def _fake_session():
    yield None


def _install_fake_control_plane(monkeypatch, *, auth_settings, identities, memberships):
    """`auth_settings`: {tenant_id: (issuer, suspended)} — a missing key means no control-plane
    row. `identities`: {(issuer, subject): identity_id}. `memberships`: {(tenant_id,
    identity_id): role}."""

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

    monkeypatch.setattr(token_verifier_module, "control_session", _fake_session)
    monkeypatch.setattr(token_verifier_module, "tenant_session", lambda ctx: _fake_session())
    monkeypatch.setattr(
        token_verifier_module, "TenantAuthSettingsRepository", FakeTenantAuthSettingsRepository
    )
    monkeypatch.setattr(token_verifier_module, "IdentityRepository", FakeIdentityRepository)
    monkeypatch.setattr(token_verifier_module, "MembershipRepository", FakeMembershipRepository)


async def test_bad_signature_is_invalid_or_expired(monkeypatch):
    tenant_id = uuid.uuid4()
    _install_fake_control_plane(
        monkeypatch,
        auth_settings={tenant_id: (ISSUER, False)},
        identities={(ISSUER, "sub-1"): uuid.uuid4()},
        memberships={},
    )
    token = _make_token(secret="a-completely-different-secret-32-bytes!", audience=str(tenant_id))
    with pytest.raises(TenantTokenVerificationError) as exc_info:
        await verify_tenant_token(
            token,
            tenant_id=tenant_id,
            key_source=_key_source,
            default_issuer=None,
            algorithm_source=lambda issuer: ("HS256",),
        )
    assert exc_info.value.reason is VerificationFailureReason.INVALID_OR_EXPIRED


async def test_expired_token_is_invalid_or_expired(monkeypatch):
    tenant_id = uuid.uuid4()
    _install_fake_control_plane(
        monkeypatch,
        auth_settings={tenant_id: (ISSUER, False)},
        identities={(ISSUER, "sub-1"): uuid.uuid4()},
        memberships={},
    )
    token = _make_token(audience=str(tenant_id), exp_delta=-60.0)
    with pytest.raises(TenantTokenVerificationError) as exc_info:
        await verify_tenant_token(
            token,
            tenant_id=tenant_id,
            key_source=_key_source,
            default_issuer=None,
            algorithm_source=lambda issuer: ("HS256",),
        )
    assert exc_info.value.reason is VerificationFailureReason.INVALID_OR_EXPIRED


async def test_no_issuer_configured_is_invalid_or_expired(monkeypatch):
    tenant_id = uuid.uuid4()
    _install_fake_control_plane(monkeypatch, auth_settings={}, identities={}, memberships={})
    token = _make_token(audience=str(tenant_id))
    with pytest.raises(TenantTokenVerificationError) as exc_info:
        await verify_tenant_token(
            token,
            tenant_id=tenant_id,
            key_source=_key_source,
            default_issuer=None,
            algorithm_source=lambda issuer: ("HS256",),
        )
    assert exc_info.value.reason is VerificationFailureReason.INVALID_OR_EXPIRED
    assert exc_info.value.issuer is None


async def test_wrong_audience_is_audience_mismatch(monkeypatch):
    tenant_id = uuid.uuid4()
    other_tenant = uuid.uuid4()
    identity_id = uuid.uuid4()
    _install_fake_control_plane(
        monkeypatch,
        auth_settings={tenant_id: (ISSUER, False)},
        identities={(ISSUER, "sub-1"): identity_id},
        memberships={(tenant_id, identity_id): "member"},
    )
    token = _make_token(audience=str(other_tenant))
    with pytest.raises(TenantTokenVerificationError) as exc_info:
        await verify_tenant_token(
            token,
            tenant_id=tenant_id,
            key_source=_key_source,
            default_issuer=None,
            algorithm_source=lambda issuer: ("HS256",),
        )
    assert exc_info.value.reason is VerificationFailureReason.AUDIENCE_MISMATCH


async def test_unknown_identity_is_unknown_identity(monkeypatch):
    tenant_id = uuid.uuid4()
    _install_fake_control_plane(
        monkeypatch,
        auth_settings={tenant_id: (ISSUER, False)},
        identities={},
        memberships={},
    )
    token = _make_token(audience=str(tenant_id))
    with pytest.raises(TenantTokenVerificationError) as exc_info:
        await verify_tenant_token(
            token,
            tenant_id=tenant_id,
            key_source=_key_source,
            default_issuer=None,
            algorithm_source=lambda issuer: ("HS256",),
        )
    assert exc_info.value.reason is VerificationFailureReason.UNKNOWN_IDENTITY


async def test_no_membership_is_missing_membership(monkeypatch):
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    _install_fake_control_plane(
        monkeypatch,
        auth_settings={tenant_id: (ISSUER, False)},
        identities={(ISSUER, "sub-1"): identity_id},
        memberships={},
    )
    token = _make_token(audience=str(tenant_id))
    with pytest.raises(TenantTokenVerificationError) as exc_info:
        await verify_tenant_token(
            token,
            tenant_id=tenant_id,
            key_source=_key_source,
            default_issuer=None,
            algorithm_source=lambda issuer: ("HS256",),
        )
    assert exc_info.value.reason is VerificationFailureReason.MISSING_MEMBERSHIP


async def test_success_resolves_identity_and_role(monkeypatch):
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    _install_fake_control_plane(
        monkeypatch,
        auth_settings={tenant_id: (ISSUER, False)},
        identities={(ISSUER, "sub-1"): identity_id},
        memberships={(tenant_id, identity_id): "admin"},
    )
    token = _make_token(audience=str(tenant_id))
    resolved = await verify_tenant_token(
        token,
        tenant_id=tenant_id,
        key_source=_key_source,
        default_issuer=None,
        algorithm_source=lambda issuer: ("HS256",),
    )
    assert resolved.identity_id == identity_id
    assert resolved.role == "admin"
    assert resolved.issuer == ISSUER


AGENT_SIGNING_KEY = "agent-token-signing-key-at-least-32-bytes-long"


def _agent_key_source(issuer: str, kid: str | None) -> str:
    """A caller (app/deps.py::get_key_source, app/mcp/server.py) that routes on issuer -- a
    minimal stand-in proving `verify_tenant_token` really does pass `AGENT_IDENTITY_ISSUER`
    through to the key source for an agent token, distinct from the human-issuer key."""
    if issuer == AGENT_IDENTITY_ISSUER:
        return AGENT_SIGNING_KEY
    return SECRET


async def test_agent_issued_token_bypasses_tenant_auth_settings(monkeypatch):
    """Gap fix (Spec 6 / #49): an agent identity's token (iss == AGENT_IDENTITY_ISSUER) is
    verified without ever consulting the tenant's own auth settings -- proven here by making that
    repository raise if it is called at all."""
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()

    class ExplodingTenantAuthSettingsRepository:
        async def get(self, session, *, tenant_id, default_issuer=None):
            raise AssertionError("must not be consulted for an agent-issuer token")

    class FakeIdentityRepository:
        async def find_by_issuer_and_subject(self, session, *, issuer, subject):
            if (issuer, subject) != (AGENT_IDENTITY_ISSUER, "agent-sub-1"):
                return None
            return SimpleNamespace(id=identity_id, issuer=issuer, subject=subject)

    class FakeMembershipRepository:
        async def get_role(self, session, ctx: RequestContext, *, identity_id):
            return "agent"

    monkeypatch.setattr(token_verifier_module, "control_session", _fake_session)
    monkeypatch.setattr(token_verifier_module, "tenant_session", lambda ctx: _fake_session())
    monkeypatch.setattr(
        token_verifier_module,
        "TenantAuthSettingsRepository",
        ExplodingTenantAuthSettingsRepository,
    )
    monkeypatch.setattr(token_verifier_module, "IdentityRepository", FakeIdentityRepository)
    monkeypatch.setattr(token_verifier_module, "MembershipRepository", FakeMembershipRepository)

    token = mint_token(
        subject="agent-sub-1",
        issuer=AGENT_IDENTITY_ISSUER,
        audience=str(tenant_id),
        signing_key=AGENT_SIGNING_KEY,
        algorithm="HS256",
        ttl_seconds=300,
        extra_claims={"cred": "agt_xyz"},
    )

    resolved = await verify_tenant_token(
        token,
        tenant_id=tenant_id,
        key_source=_agent_key_source,
        default_issuer=None,
        algorithm_source=lambda issuer: ("HS256",),
    )

    assert resolved.identity_id == identity_id
    assert resolved.role == "agent"
    assert resolved.issuer == AGENT_IDENTITY_ISSUER
    assert resolved.credential_public_id == "agt_xyz"


async def test_agent_issued_token_still_fails_closed_on_a_bad_signature(monkeypatch):
    """The unverified issuer peek is never trusted on its own: a token claiming
    AGENT_IDENTITY_ISSUER but signed with the wrong key still fails verification."""
    tenant_id = uuid.uuid4()

    class ExplodingTenantAuthSettingsRepository:
        async def get(self, session, *, tenant_id, default_issuer=None):
            raise AssertionError("must not be consulted for an agent-issuer token")

    monkeypatch.setattr(
        token_verifier_module,
        "TenantAuthSettingsRepository",
        ExplodingTenantAuthSettingsRepository,
    )

    token = mint_token(
        subject="agent-sub-1",
        issuer=AGENT_IDENTITY_ISSUER,
        audience=str(tenant_id),
        signing_key="a-completely-different-signing-key-32-bytes",
        algorithm="HS256",
        ttl_seconds=300,
    )

    with pytest.raises(TenantTokenVerificationError) as exc_info:
        await verify_tenant_token(
            token,
            tenant_id=tenant_id,
            key_source=_agent_key_source,
            default_issuer=None,
            algorithm_source=lambda issuer: ("HS256",),
        )
    assert exc_info.value.reason is VerificationFailureReason.INVALID_OR_EXPIRED


async def test_success_is_not_affected_by_tenant_suspension(monkeypatch):
    """The shared module does not check suspension at all (issue #69 owns that check, in each
    caller) -- a suspended tenant's otherwise-valid token still resolves here."""
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    _install_fake_control_plane(
        monkeypatch,
        auth_settings={tenant_id: (ISSUER, True)},
        identities={(ISSUER, "sub-1"): identity_id},
        memberships={(tenant_id, identity_id): "member"},
    )
    token = _make_token(audience=str(tenant_id))
    resolved = await verify_tenant_token(
        token,
        tenant_id=tenant_id,
        key_source=_key_source,
        default_issuer=None,
        algorithm_source=lambda issuer: ("HS256",),
    )
    assert resolved.identity_id == identity_id
