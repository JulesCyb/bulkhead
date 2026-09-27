"""Unit tests for `app.agent_credential_exchange.derive_public_key_pem` (spec A4 / #94, #112):
the pure PEM private-key -> public-key derivation relocated out of
`app.config.Settings._require_consistent_agent_token_key`, which now only *checks* consistency
by calling it. `tests/test_agent_token_algorithm_config.py` still covers that Settings-level
check (asymmetric algorithm, derived/explicit verification key, malformed key -- all through
`Settings(...)`); this file tests the derivation function itself, directly, with no `Settings`
object involved.
"""

from __future__ import annotations

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from app.agent_credential_exchange import AgentTokenKeyDerivationError, derive_public_key_pem


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


def test_derive_public_key_pem_returns_the_matching_public_key():
    private_pem, public_pem = _rsa_keypair()
    assert derive_public_key_pem(private_pem) == public_pem


def test_derive_public_key_pem_rejects_a_malformed_key():
    with pytest.raises(AgentTokenKeyDerivationError, match="PEM-encoded private key"):
        derive_public_key_pem("not-a-pem-key")


def test_derive_public_key_pem_rejects_a_plain_shared_secret():
    # A plain HS-style shared secret is not a PEM private key -- must fail closed rather than
    # silently succeed with an unusable configuration.
    with pytest.raises(AgentTokenKeyDerivationError, match="PEM-encoded private key"):
        derive_public_key_pem("a-plain-shared-secret-32-bytes-long")
