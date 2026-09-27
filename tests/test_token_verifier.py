"""Unit tests for the shared tenant-token verifier (#44): exercises `verify_tenant_token` as a
black box, with the shared `FakeControlPlaneReads` (#100) passed explicitly as its `adapter=`
keyword -- no monkeypatching of any name inside `app.token_verifier` itself.
"""

from __future__ import annotations

import time
import uuid

import jwt
import pytest

from app.jwt_verifier import mint_token
from app.token_verifier import (
    AGENT_IDENTITY_ISSUER,
    TenantSuspendedAtVerification,
    TenantTokenVerificationError,
    VerificationFailureReason,
    verify_tenant_token,
)
from tests.conftest import FakeControlPlaneReads

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


async def test_bad_signature_is_invalid_or_expired():
    tenant_id = uuid.uuid4()
    adapter = FakeControlPlaneReads(
        auth_settings={tenant_id: (ISSUER, False)},
        identities={(ISSUER, "sub-1"): uuid.uuid4()},
    )
    token = _make_token(secret="a-completely-different-secret-32-bytes!", audience=str(tenant_id))
    with pytest.raises(TenantTokenVerificationError) as exc_info:
        await verify_tenant_token(
            token,
            tenant_id=tenant_id,
            key_source=_key_source,
            default_issuer=None,
            algorithm_source=lambda issuer: ("HS256",),
            adapter=adapter,
        )
    assert exc_info.value.reason is VerificationFailureReason.INVALID_OR_EXPIRED


async def test_expired_token_is_invalid_or_expired():
    tenant_id = uuid.uuid4()
    adapter = FakeControlPlaneReads(
        auth_settings={tenant_id: (ISSUER, False)},
        identities={(ISSUER, "sub-1"): uuid.uuid4()},
    )
    token = _make_token(audience=str(tenant_id), exp_delta=-60.0)
    with pytest.raises(TenantTokenVerificationError) as exc_info:
        await verify_tenant_token(
            token,
            tenant_id=tenant_id,
            key_source=_key_source,
            default_issuer=None,
            algorithm_source=lambda issuer: ("HS256",),
            adapter=adapter,
        )
    assert exc_info.value.reason is VerificationFailureReason.INVALID_OR_EXPIRED


async def test_no_issuer_configured_is_invalid_or_expired():
    tenant_id = uuid.uuid4()
    adapter = FakeControlPlaneReads()
    token = _make_token(audience=str(tenant_id))
    with pytest.raises(TenantTokenVerificationError) as exc_info:
        await verify_tenant_token(
            token,
            tenant_id=tenant_id,
            key_source=_key_source,
            default_issuer=None,
            algorithm_source=lambda issuer: ("HS256",),
            adapter=adapter,
        )
    assert exc_info.value.reason is VerificationFailureReason.INVALID_OR_EXPIRED
    assert exc_info.value.issuer is None


async def test_wrong_audience_is_audience_mismatch():
    tenant_id = uuid.uuid4()
    other_tenant = uuid.uuid4()
    identity_id = uuid.uuid4()
    adapter = FakeControlPlaneReads(
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
            adapter=adapter,
        )
    assert exc_info.value.reason is VerificationFailureReason.AUDIENCE_MISMATCH


async def test_unknown_identity_is_unknown_identity():
    tenant_id = uuid.uuid4()
    adapter = FakeControlPlaneReads(auth_settings={tenant_id: (ISSUER, False)})
    token = _make_token(audience=str(tenant_id))
    with pytest.raises(TenantTokenVerificationError) as exc_info:
        await verify_tenant_token(
            token,
            tenant_id=tenant_id,
            key_source=_key_source,
            default_issuer=None,
            algorithm_source=lambda issuer: ("HS256",),
            adapter=adapter,
        )
    assert exc_info.value.reason is VerificationFailureReason.UNKNOWN_IDENTITY


async def test_no_membership_is_missing_membership():
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    adapter = FakeControlPlaneReads(
        auth_settings={tenant_id: (ISSUER, False)},
        identities={(ISSUER, "sub-1"): identity_id},
    )
    token = _make_token(audience=str(tenant_id))
    with pytest.raises(TenantTokenVerificationError) as exc_info:
        await verify_tenant_token(
            token,
            tenant_id=tenant_id,
            key_source=_key_source,
            default_issuer=None,
            algorithm_source=lambda issuer: ("HS256",),
            adapter=adapter,
        )
    assert exc_info.value.reason is VerificationFailureReason.MISSING_MEMBERSHIP


async def test_success_resolves_identity_and_role():
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    adapter = FakeControlPlaneReads(
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
        adapter=adapter,
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


async def test_agent_issued_token_bypasses_tenant_auth_settings():
    """Gap fix (Spec 6 / #49): an agent identity's token (iss == AGENT_IDENTITY_ISSUER) is
    verified without ever consulting the tenant's own auth settings -- proven here by making the
    fake adapter raise if `get_tenant_auth_settings` is called at all."""
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    adapter = FakeControlPlaneReads(
        identities={(AGENT_IDENTITY_ISSUER, "agent-sub-1"): identity_id},
        memberships={(tenant_id, identity_id): "agent"},
        explode=frozenset({"get_tenant_auth_settings"}),
    )

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
        adapter=adapter,
    )

    assert resolved.identity_id == identity_id
    assert resolved.role == "agent"
    assert resolved.issuer == AGENT_IDENTITY_ISSUER
    assert resolved.credential_public_id == "agt_xyz"


async def test_agent_issued_token_still_fails_closed_on_a_bad_signature():
    """The unverified issuer peek is never trusted on its own: a token claiming
    AGENT_IDENTITY_ISSUER but signed with the wrong key still fails verification."""
    tenant_id = uuid.uuid4()
    adapter = FakeControlPlaneReads(explode=frozenset({"get_tenant_auth_settings"}))

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
            adapter=adapter,
        )
    assert exc_info.value.reason is VerificationFailureReason.INVALID_OR_EXPIRED


async def test_a_suspended_tenant_is_refused_before_any_identity_or_membership_lookup():
    """Code review 2026-09-26 retired the `read_tenant_record` opt-out: the verifier always reads
    the tenant record after the audience check and refuses a suspended tenant there -- an
    otherwise-valid token never reaches the identity or membership lookup."""
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    adapter = FakeControlPlaneReads(
        auth_settings={tenant_id: (ISSUER, True)},
        identities={(ISSUER, "sub-1"): identity_id},
        memberships={(tenant_id, identity_id): "member"},
        explode=frozenset({"find_identity_by_issuer_and_subject", "get_membership_role"}),
    )
    token = _make_token(audience=str(tenant_id))
    with pytest.raises(TenantSuspendedAtVerification) as exc_info:
        await verify_tenant_token(
            token,
            tenant_id=tenant_id,
            key_source=_key_source,
            default_issuer=None,
            algorithm_source=lambda issuer: ("HS256",),
            adapter=adapter,
        )
    assert exc_info.value.tenant_id == tenant_id
    assert exc_info.value.issuer == ISSUER


async def test_an_unsuspended_tenant_record_is_returned_on_the_resolved_identity():
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    adapter = FakeControlPlaneReads(
        auth_settings={tenant_id: (ISSUER, False)},
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
        adapter=adapter,
    )
    assert resolved.identity_id == identity_id
    assert resolved.tenant_record is not None
    assert resolved.tenant_record.tenant_id == tenant_id
