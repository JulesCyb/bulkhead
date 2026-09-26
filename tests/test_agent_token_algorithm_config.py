"""Unit tests for the algorithm-confusion guard at `Settings` construction (review finding, Spec
6, ADR-0005, ADR-0003): `jwt_algorithm` (human tokens, verified against `jwt_verification_key`)
and `agent_token_algorithm` (agent tokens, verified against `agent_token_signing_key` /
`agent_token_verification_key`) are independent settings on purpose -- these tests prove the
fail-closed checks around them construct a `Settings` object that later fails to mint or verify
anything unsafe, before any request or tool call is ever accepted.
"""

from __future__ import annotations

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import ValidationError

from app.config import Settings
from app.jwt_verifier import MIN_HS_SECRET_BYTES

_BASE_FIELDS = dict(
    _env_file=None,
    environment="test",
    auth_mode="jwt",
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
)


def _settings(**overrides) -> Settings:
    fields = dict(_BASE_FIELDS)
    fields.update(overrides)
    return Settings(**fields)


# --- Supported-algorithm allow-list: "none" and typos are refused for either setting ---


@pytest.mark.parametrize("field", ["jwt_algorithm", "agent_token_algorithm"])
def test_none_algorithm_is_refused_for_either_setting(field):
    with pytest.raises(ValidationError, match="not a supported signing algorithm"):
        _settings(**{field: "none"})


@pytest.mark.parametrize("field", ["jwt_algorithm", "agent_token_algorithm"])
def test_unknown_algorithm_is_refused_for_either_setting(field):
    with pytest.raises(ValidationError, match="not a supported signing algorithm"):
        _settings(**{field: "not-a-real-algorithm"})


def test_default_algorithms_are_independent_and_supported():
    settings = _settings()
    assert settings.jwt_algorithm == "RS256"
    assert settings.agent_token_algorithm == "HS256"


# --- HS* agent_token_signing_key must be long enough to resist brute force ---


def test_short_hs_agent_signing_key_fails_startup():
    with pytest.raises(ValidationError, match="too short"):
        _settings(agent_token_algorithm="HS256", agent_token_signing_key="short-secret")


def test_minimum_length_hs_agent_signing_key_is_accepted():
    secret = "x" * MIN_HS_SECRET_BYTES
    settings = _settings(agent_token_algorithm="HS256", agent_token_signing_key=secret)
    assert settings.agent_token_signing_key.get_secret_value() == secret


def test_unconfigured_agent_signing_key_does_not_raise():
    # No key configured at all is a valid (if unusable-for-agents) deployment -- validated only
    # when the key is actually present; the exchange endpoint itself fails closed at request time
    # (app/agent_credential_exchange.py) for the unconfigured case.
    settings = _settings(agent_token_signing_key=None)
    assert settings.agent_token_signing_key is None


# --- Asymmetric agent_token_algorithm: signing key must be a private key; the public half is
# derived automatically when not set explicitly ---


def _rsa_keypair() -> tuple[str, str]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    public_pem = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    return private_pem, public_pem


def test_asymmetric_agent_algorithm_derives_the_verification_key_when_unset():
    private_pem, public_pem = _rsa_keypair()
    settings = _settings(agent_token_algorithm="RS256", agent_token_signing_key=private_pem)
    assert settings.agent_token_verification_key is not None
    assert settings.agent_token_verification_key.get_secret_value() == public_pem


def test_asymmetric_agent_algorithm_keeps_an_explicit_verification_key():
    private_pem, public_pem = _rsa_keypair()
    other_private_pem, other_public_pem = _rsa_keypair()
    settings = _settings(
        agent_token_algorithm="RS256",
        agent_token_signing_key=private_pem,
        agent_token_verification_key=other_public_pem,
    )
    # Explicit value wins -- never silently overwritten by the derived one.
    assert settings.agent_token_verification_key.get_secret_value() == other_public_pem
    assert settings.agent_token_verification_key.get_secret_value() != public_pem


def test_asymmetric_agent_algorithm_with_a_malformed_signing_key_fails_startup():
    with pytest.raises(ValidationError, match="PEM-encoded private key"):
        _settings(agent_token_algorithm="RS256", agent_token_signing_key="not-a-pem-key")


def test_asymmetric_agent_algorithm_rejects_a_shared_secret_as_the_signing_key():
    # A plain HS-style shared secret is not a PEM private key -- must fail closed rather than
    # silently succeed with an unusable configuration.
    with pytest.raises(ValidationError, match="PEM-encoded private key"):
        _settings(
            agent_token_algorithm="ES256",
            agent_token_signing_key="a-plain-shared-secret-32-bytes-long",
        )
