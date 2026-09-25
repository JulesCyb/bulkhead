"""ASGI-seam tests for the agent-credential token exchange (#47, ADR-0005, ADR-0012).

`AgentCredentialRepository`, `IdentityRepository`, and `TenantAuthSettingsRepository` are faked
here (no real Postgres) -- the same pattern `tests/test_jwt_auth.py` and
`tests/test_memberships_api.py` use; the real credential table, its RLS, and its hashing/timing
guarantees have their own coverage in `tests/test_agent_credentials_integration.py`. This file
proves the HTTP-facing contract: a valid exchange yields a tenant-scoped, agent-subject token;
every failure kind is indistinguishable; a successful exchange (and only a successful one)
advances the credential's last-used marker; and the minted token really does verify through the
shared `token_verifier` module and resolve back to the same identity and tenant.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import jwt
import pytest

import app.agent_credential_exchange as exchange_module
import app.token_verifier as token_verifier_module
from app.config import Settings, get_settings
from app.context import RequestContext
from app.main import app

SIGNING_KEY = "agent-token-exchange-test-shared-secret-32-bytes"
ISSUER = "https://idp.example.com"


@asynccontextmanager
async def _fake_session():
    yield None


class FakeCredentialStore:
    """A tiny in-memory stand-in for the `agent_credentials` table, precise enough to prove the
    last-used-only-on-success invariant at the ASGI layer without a real database.

    `credentials`: public_id -> {"secret", "identity_id", "tenant_id", "revoked", "last_used"}.
    """

    def __init__(self, credentials: dict[str, dict]) -> None:
        self.credentials = credentials

    async def verify_and_touch(self, session, ctx: RequestContext, *, public_id: str, secret: str):
        row = self.credentials.get(public_id)
        if row is None or row["tenant_id"] != ctx.tenant_id:
            return None
        if row["revoked"] or row["secret"] != secret:
            return None
        row["last_used"] = "touched"
        return SimpleNamespace(
            id=uuid.uuid4(), identity_id=row["identity_id"], tenant_id=ctx.tenant_id
        )


def _install_fakes(
    monkeypatch,
    *,
    store: FakeCredentialStore,
    identities: dict[uuid.UUID, tuple[str, str]],
    auth_settings: dict[uuid.UUID, str | None],
):
    """`identities`: identity_id -> (issuer, subject). `auth_settings`: tenant_id -> issuer (or
    None for "no control-plane row", falling back to the default issuer, mirroring the real
    repository)."""

    class FakeAgentCredentialRepository:
        def __call__(self):
            return self

        async def verify_and_touch(self, *args, **kwargs):
            return await store.verify_and_touch(*args, **kwargs)

    class FakeIdentityRepository:
        async def get_by_id(self, session, *, identity_id):
            pair = identities.get(identity_id)
            if pair is None:
                return None
            issuer, subject = pair
            return SimpleNamespace(id=identity_id, issuer=issuer, subject=subject)

    class FakeTenantAuthSettingsRepository:
        async def get(self, session, *, tenant_id, default_issuer=None):
            if tenant_id not in auth_settings:
                return None
            issuer = auth_settings[tenant_id] or default_issuer
            return SimpleNamespace(issuer=issuer, suspended=False)

    monkeypatch.setattr(exchange_module, "tenant_session", lambda ctx: _fake_session())
    monkeypatch.setattr(exchange_module, "control_session", _fake_session)
    monkeypatch.setattr(
        exchange_module, "AgentCredentialRepository", FakeAgentCredentialRepository()
    )
    monkeypatch.setattr(exchange_module, "IdentityRepository", FakeIdentityRepository)
    monkeypatch.setattr(
        exchange_module, "TenantAuthSettingsRepository", FakeTenantAuthSettingsRepository
    )


def _settings(**overrides) -> Settings:
    fields = dict(
        _env_file=None,
        environment="test",
        auth_mode="jwt",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        default_identity_issuer=None,
        jwt_verification_key=SIGNING_KEY,
        jwt_algorithm="HS256",
        agent_token_signing_key=SIGNING_KEY,
        agent_token_ttl_seconds=300,
    )
    fields.update(overrides)
    return Settings(**fields)


@pytest.fixture
def client_settings(monkeypatch):
    settings = _settings()
    app.dependency_overrides[get_settings] = lambda: settings
    yield settings
    app.dependency_overrides.pop(get_settings, None)


async def _exchange(tenant_id: uuid.UUID, public_id: str, secret: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            f"/v1/t/{tenant_id}/agent-tokens", json={"public_id": public_id, "secret": secret}
        )


# --- AC1: a valid, unrevoked credential yields a token naming the agent identity and the tenant ---


async def test_valid_credential_yields_a_token_scoped_to_its_own_tenant(
    monkeypatch, client_settings
):
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    store = FakeCredentialStore(
        {
            "agt_abc": {
                "secret": "correct-secret",
                "identity_id": identity_id,
                "tenant_id": tenant_id,
                "revoked": False,
                "last_used": None,
            }
        }
    )
    _install_fakes(
        monkeypatch,
        store=store,
        identities={identity_id: (ISSUER, "agent-sub-1")},
        auth_settings={tenant_id: ISSUER},
    )

    response = await _exchange(tenant_id, "agt_abc", "correct-secret")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 300
    claims = jwt.decode(
        body["access_token"],
        SIGNING_KEY,
        algorithms=["HS256"],
        issuer=ISSUER,
        audience=str(tenant_id),
    )
    assert claims["sub"] == "agent-sub-1"
    assert claims["aud"] == str(tenant_id)
    assert claims["iss"] == ISSUER


# --- AC2: unknown identifier, wrong secret, and revoked all fail exactly the same way ---


@pytest.mark.parametrize(
    "public_id,secret",
    [
        ("agt_unknown", "whatever"),
        ("agt_real", "wrong-secret"),
        ("agt_revoked", "revoked-secret"),
    ],
    ids=["unknown-identifier", "wrong-secret", "revoked-credential"],
)
async def test_every_failure_kind_produces_the_identical_response(
    monkeypatch, client_settings, public_id, secret
):
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    store = FakeCredentialStore(
        {
            "agt_real": {
                "secret": "the-real-secret",
                "identity_id": identity_id,
                "tenant_id": tenant_id,
                "revoked": False,
                "last_used": None,
            },
            "agt_revoked": {
                "secret": "revoked-secret",
                "identity_id": identity_id,
                "tenant_id": tenant_id,
                "revoked": True,
                "last_used": None,
            },
        }
    )
    _install_fakes(
        monkeypatch,
        store=store,
        identities={identity_id: (ISSUER, "agent-sub-1")},
        auth_settings={tenant_id: ISSUER},
    )

    response = await _exchange(tenant_id, public_id, secret)

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid credential."}


async def test_all_three_failure_responses_are_byte_identical_to_each_other(
    monkeypatch, client_settings
):
    """Not just the same status/shape per case above -- the three responses are indistinguishable
    from one another, checked directly against each other rather than against a fixed literal."""
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    store = FakeCredentialStore(
        {
            "agt_real": {
                "secret": "the-real-secret",
                "identity_id": identity_id,
                "tenant_id": tenant_id,
                "revoked": False,
                "last_used": None,
            },
            "agt_revoked": {
                "secret": "revoked-secret",
                "identity_id": identity_id,
                "tenant_id": tenant_id,
                "revoked": True,
                "last_used": None,
            },
        }
    )
    _install_fakes(
        monkeypatch,
        store=store,
        identities={identity_id: (ISSUER, "agent-sub-1")},
        auth_settings={tenant_id: ISSUER},
    )

    unknown = await _exchange(tenant_id, "agt_never_issued", "anything")
    wrong_secret = await _exchange(tenant_id, "agt_real", "not-the-real-secret")
    revoked = await _exchange(tenant_id, "agt_revoked", "revoked-secret")

    bodies = {r.status_code for r in (unknown, wrong_secret, revoked)}
    assert bodies == {401}
    assert unknown.json() == wrong_secret.json() == revoked.json()


# --- AC3: a successful exchange advances last-used; a failed one never does ---


async def test_last_used_advances_only_on_success(monkeypatch, client_settings):
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    store = FakeCredentialStore(
        {
            "agt_abc": {
                "secret": "correct-secret",
                "identity_id": identity_id,
                "tenant_id": tenant_id,
                "revoked": False,
                "last_used": None,
            }
        }
    )
    _install_fakes(
        monkeypatch,
        store=store,
        identities={identity_id: (ISSUER, "agent-sub-1")},
        auth_settings={tenant_id: ISSUER},
    )

    failed = await _exchange(tenant_id, "agt_abc", "wrong-secret")
    assert failed.status_code == 401
    assert store.credentials["agt_abc"]["last_used"] is None

    succeeded = await _exchange(tenant_id, "agt_abc", "correct-secret")
    assert succeeded.status_code == 200
    assert store.credentials["agt_abc"]["last_used"] == "touched"


# --- AC4: the issued token round-trips through the shared verifier back to the same identity ---


async def test_issued_token_round_trips_through_the_shared_verifier(monkeypatch, client_settings):
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    store = FakeCredentialStore(
        {
            "agt_abc": {
                "secret": "correct-secret",
                "identity_id": identity_id,
                "tenant_id": tenant_id,
                "revoked": False,
                "last_used": None,
            }
        }
    )
    _install_fakes(
        monkeypatch,
        store=store,
        identities={identity_id: (ISSUER, "agent-sub-1")},
        auth_settings={tenant_id: ISSUER},
    )

    issued = await _exchange(tenant_id, "agt_abc", "correct-secret")
    assert issued.status_code == 200
    token = issued.json()["access_token"]

    # Verify through the exact module Spec 6's MCP transport is meant to reuse (#44), not by
    # re-decoding the JWT by hand -- this is the round trip the acceptance criterion asks for.
    def _key_source(issuer: str, kid: str | None) -> str:
        return SIGNING_KEY

    class FakeIdentityRepositoryForVerify:
        async def find_by_issuer_and_subject(self, session, *, issuer, subject):
            if (issuer, subject) != (ISSUER, "agent-sub-1"):
                return None
            return SimpleNamespace(id=identity_id, issuer=issuer, subject=subject)

    class FakeMembershipRepository:
        async def get_role(self, session, ctx: RequestContext, *, identity_id):
            return "agent"

    class FakeTenantAuthSettingsRepositoryForVerify:
        async def get(self, session, *, tenant_id, default_issuer=None):
            return SimpleNamespace(issuer=ISSUER, suspended=False)

    monkeypatch.setattr(token_verifier_module, "control_session", _fake_session)
    monkeypatch.setattr(token_verifier_module, "tenant_session", lambda ctx: _fake_session())
    monkeypatch.setattr(
        token_verifier_module, "IdentityRepository", FakeIdentityRepositoryForVerify
    )
    monkeypatch.setattr(token_verifier_module, "MembershipRepository", FakeMembershipRepository)
    monkeypatch.setattr(
        token_verifier_module,
        "TenantAuthSettingsRepository",
        FakeTenantAuthSettingsRepositoryForVerify,
    )

    resolved = await token_verifier_module.verify_tenant_token(
        token,
        tenant_id=tenant_id,
        key_source=_key_source,
        default_issuer=None,
        algorithms=("HS256",),
    )

    assert resolved.identity_id == identity_id
    assert resolved.role == "agent"


# --- Misconfiguration fails closed rather than minting an unusable or unsigned token ---


async def test_missing_signing_key_fails_the_exchange(monkeypatch):
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    store = FakeCredentialStore(
        {
            "agt_abc": {
                "secret": "correct-secret",
                "identity_id": identity_id,
                "tenant_id": tenant_id,
                "revoked": False,
                "last_used": None,
            }
        }
    )
    _install_fakes(
        monkeypatch,
        store=store,
        identities={identity_id: (ISSUER, "agent-sub-1")},
        auth_settings={tenant_id: ISSUER},
    )
    settings = _settings(agent_token_signing_key=None)
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        response = await _exchange(tenant_id, "agt_abc", "correct-secret")
    finally:
        app.dependency_overrides.pop(get_settings, None)

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid credential."}
