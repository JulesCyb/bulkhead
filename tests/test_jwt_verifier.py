"""Unit tests for the isolated JWT-verification component (#24): signature, issuer, and expiry
checks only -- no database, no network. Every test mints its own HS256 token with an in-memory
key source; nothing here ever imports app.deps or touches a session."""

from __future__ import annotations

import time

import jwt
import pytest

from app.jwt_verifier import TokenVerificationError, verify_token

SECRET = "unit-test-shared-secret-at-least-32-bytes-long"
ISSUER = "https://idp.example.com"


def _key_source(issuer: str, kid: str | None) -> str:
    return SECRET


def _make_token(
    *,
    secret: str = SECRET,
    issuer: str = ISSUER,
    subject: str = "sub-123",
    audience: str = "tenant-abc",
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


def test_verify_token_accepts_a_well_formed_token():
    token = _make_token()
    claims = verify_token(
        token, key_source=_key_source, expected_issuer=ISSUER, algorithms=("HS256",)
    )
    assert claims.issuer == ISSUER
    assert claims.subject == "sub-123"
    assert claims.audience == "tenant-abc"


def test_verify_token_rejects_a_bad_signature():
    token = _make_token(secret="wrong-secret-but-also-at-least-32-bytes-long")
    with pytest.raises(TokenVerificationError):
        verify_token(token, key_source=_key_source, expected_issuer=ISSUER, algorithms=("HS256",))


def test_verify_token_rejects_an_expired_token():
    token = _make_token(exp_delta=-10.0)
    with pytest.raises(TokenVerificationError):
        verify_token(token, key_source=_key_source, expected_issuer=ISSUER, algorithms=("HS256",))


def test_verify_token_rejects_an_unexpected_issuer():
    token = _make_token(issuer="https://someone-else.example.com")
    with pytest.raises(TokenVerificationError):
        verify_token(token, key_source=_key_source, expected_issuer=ISSUER, algorithms=("HS256",))


def test_verify_token_rejects_a_malformed_token():
    with pytest.raises(TokenVerificationError):
        verify_token(
            "not-a-jwt", key_source=_key_source, expected_issuer=ISSUER, algorithms=("HS256",)
        )


def test_verify_token_rejects_when_key_source_has_no_key():
    def _no_key(issuer: str, kid: str | None) -> str:
        raise LookupError("no key for issuer")

    token = _make_token()
    with pytest.raises(TokenVerificationError):
        verify_token(token, key_source=_no_key, expected_issuer=ISSUER, algorithms=("HS256",))


def test_verify_token_does_not_check_audience():
    """Audience is the caller's job (app/deps.py, against the URL path) -- the verifier returns
    whatever audience the token carries without judging it."""
    token = _make_token(audience="some-other-tenant")
    claims = verify_token(
        token, key_source=_key_source, expected_issuer=ISSUER, algorithms=("HS256",)
    )
    assert claims.audience == "some-other-tenant"
