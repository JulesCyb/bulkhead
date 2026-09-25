"""ASGI-seam tests for AUTH_MODE=jwt (#24, ADR-0003, ADR-0012): table-driven, one case per
branch. The control-plane and membership repositories app/deps.py calls are faked here (no real
Postgres) so these run fast and exercise exactly the ASGI + dependency wiring; the real
migrations, RLS, and grants around those repositories already have their own coverage in
tests/test_rls_integration.py.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import jwt
import pytest

import app.deps as deps_module
import app.tenant_suspension as tenant_suspension_module
import app.token_verifier as token_verifier_module
from app.config import Settings, get_settings
from app.context import RequestContext
from app.main import app

SECRET = "asgi-jwt-test-shared-secret-at-least-32-bytes"
ISSUER = "https://idp.example.com"


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


@asynccontextmanager
async def _fake_session():
    yield None


def _install_fake_control_plane(monkeypatch, *, auth_settings, identities, memberships):
    """`auth_settings`: {tenant_id: (issuer, suspended)} — a missing key means no control-plane
    row (falls back to the default issuer, not suspended), mirroring the real repository.
    `identities`: {(issuer, subject): identity_id}. `memberships`: {(tenant_id, identity_id): role}.

    Installed on both `app.tenant_suspension` (the one shared suspension check every
    context-resolution seam calls, issue #69) and `app.token_verifier` (the shared
    signature/audience/identity/membership check, issue #44) — `app.deps` no longer keeps its own
    copy of either.
    """

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

    monkeypatch.setattr(tenant_suspension_module, "control_session", _fake_session)
    monkeypatch.setattr(
        tenant_suspension_module,
        "TenantAuthSettingsRepository",
        FakeTenantAuthSettingsRepository,
    )
    monkeypatch.setattr(token_verifier_module, "control_session", _fake_session)
    monkeypatch.setattr(
        token_verifier_module, "TenantAuthSettingsRepository", FakeTenantAuthSettingsRepository
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


async def _post_run(tenant_id: uuid.UUID, token: str | None) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            f"/v1/t/{tenant_id}/agents/assistant/run",
            json={"prompt": "Hi"},
            headers=headers,
        )


async def test_missing_authorization_header_is_unauthenticated(jwt_client, monkeypatch):
    tenant_id = uuid.uuid4()
    _install_fake_control_plane(monkeypatch, auth_settings={}, identities={}, memberships={})
    response = await _post_run(tenant_id, token=None)
    assert response.status_code == 401


async def test_bad_signature_is_unauthenticated(jwt_client, monkeypatch):
    tenant_id = uuid.uuid4()
    _install_fake_control_plane(
        monkeypatch,
        auth_settings={tenant_id: (ISSUER, False)},
        identities={(ISSUER, "sub-1"): uuid.uuid4()},
        memberships={},
    )
    token = _make_token(secret="a-completely-different-secret-32-bytes!", audience=str(tenant_id))
    response = await _post_run(tenant_id, token)
    assert response.status_code == 401


async def test_expired_token_is_unauthenticated(jwt_client, monkeypatch):
    tenant_id = uuid.uuid4()
    _install_fake_control_plane(
        monkeypatch,
        auth_settings={tenant_id: (ISSUER, False)},
        identities={(ISSUER, "sub-1"): uuid.uuid4()},
        memberships={},
    )
    token = _make_token(audience=str(tenant_id), exp_delta=-60.0)
    response = await _post_run(tenant_id, token)
    assert response.status_code == 401


async def test_audience_naming_wrong_tenant_is_forbidden_with_generic_body(
    jwt_client, monkeypatch, caplog
):
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
    with caplog.at_level("WARNING"):
        response = await _post_run(tenant_id, token)
    assert response.status_code == 403
    assert response.json()["detail"] == deps_module.FORBIDDEN_DETAIL
    reasons = [r.reason for r in caplog.records if hasattr(r, "reason")]
    assert "audience_mismatch" in reasons


async def test_unknown_identity_is_forbidden_with_generic_body(jwt_client, monkeypatch, caplog):
    tenant_id = uuid.uuid4()
    _install_fake_control_plane(
        monkeypatch,
        auth_settings={tenant_id: (ISSUER, False)},
        identities={},
        memberships={},
    )
    token = _make_token(audience=str(tenant_id))
    with caplog.at_level("WARNING"):
        response = await _post_run(tenant_id, token)
    assert response.status_code == 403
    assert response.json()["detail"] == deps_module.FORBIDDEN_DETAIL
    reasons = [r.reason for r in caplog.records if hasattr(r, "reason")]
    assert "unknown_identity" in reasons


async def test_suspended_tenant_is_forbidden_with_generic_body(jwt_client, monkeypatch, caplog):
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    _install_fake_control_plane(
        monkeypatch,
        auth_settings={tenant_id: (ISSUER, True)},
        identities={(ISSUER, "sub-1"): identity_id},
        memberships={(tenant_id, identity_id): "member"},
    )
    token = _make_token(audience=str(tenant_id))
    with caplog.at_level("WARNING"):
        response = await _post_run(tenant_id, token)
    assert response.status_code == 403
    assert response.json()["detail"] == deps_module.FORBIDDEN_DETAIL
    reasons = [r.reason for r in caplog.records if hasattr(r, "reason")]
    assert "tenant_suspended" in reasons


async def test_missing_membership_is_forbidden_with_generic_body(jwt_client, monkeypatch, caplog):
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    _install_fake_control_plane(
        monkeypatch,
        auth_settings={tenant_id: (ISSUER, False)},
        identities={(ISSUER, "sub-1"): identity_id},
        memberships={},
    )
    token = _make_token(audience=str(tenant_id))
    with caplog.at_level("WARNING"):
        response = await _post_run(tenant_id, token)
    assert response.status_code == 403
    assert response.json()["detail"] == deps_module.FORBIDDEN_DETAIL
    reasons = [r.reason for r in caplog.records if hasattr(r, "reason")]
    assert "missing_membership" in reasons


async def test_nonexistent_tenant_gets_the_identical_forbidden_response_as_a_non_member(
    jwt_client, monkeypatch
):
    """A path naming a tenant with no control-plane row and no membership row at all (i.e. it
    was never created) is rejected exactly like a real tenant the caller isn't a member of --
    both fall through to the same missing-membership branch."""
    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    _install_fake_control_plane(
        monkeypatch,
        auth_settings={},  # no control-plane row at all
        identities={(ISSUER, "sub-1"): identity_id},
        memberships={},  # and so, naturally, no membership either
    )
    settings = Settings(
        _env_file=None,
        environment="test",
        auth_mode="jwt",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        default_identity_issuer=ISSUER,
        jwt_verification_key=SECRET,
        jwt_algorithm="HS256",
    )
    app.dependency_overrides[get_settings] = lambda: settings
    token = _make_token(audience=str(tenant_id))
    response_nonexistent = await _post_run(tenant_id, token)

    other_tenant = uuid.uuid4()
    _install_fake_control_plane(
        monkeypatch,
        auth_settings={other_tenant: (ISSUER, False)},
        identities={(ISSUER, "sub-1"): identity_id},
        memberships={},
    )
    token_2 = _make_token(audience=str(other_tenant))
    response_non_member = await _post_run(other_tenant, token_2)

    assert response_nonexistent.status_code == response_non_member.status_code == 403
    assert response_nonexistent.json() == response_non_member.json()


async def test_a_change_fed_only_into_the_shared_module_is_observed_at_the_http_layer(
    jwt_client, monkeypatch, caplog
):
    """Proves delegation (#44 acceptance criterion 3): `app/deps.py` no longer contains its own
    copy of the audience check. Patching `verify_tenant_token` itself -- the shared module's only
    entry point -- to always report a wrong-audience failure, with no other part of the control
    plane faked, is enough to make the HTTP layer reject the request. If `app/deps.py` still ran
    its own audience check, this patch alone could not produce a 403 here."""
    import app.token_verifier as token_verifier_module

    tenant_id = uuid.uuid4()

    async def _always_audience_mismatch(*args, **kwargs):
        raise token_verifier_module.TenantTokenVerificationError(
            token_verifier_module.VerificationFailureReason.AUDIENCE_MISMATCH, issuer=ISSUER
        )

    monkeypatch.setattr(deps_module, "verify_tenant_token", _always_audience_mismatch)
    token = _make_token(audience=str(tenant_id))
    with caplog.at_level("WARNING"):
        response = await _post_run(tenant_id, token)
    assert response.status_code == 403
    assert response.json()["detail"] == deps_module.FORBIDDEN_DETAIL
    reasons = [r.reason for r in caplog.records if hasattr(r, "reason")]
    assert "audience_mismatch" in reasons


async def test_valid_token_succeeds_and_roles_come_from_the_membership_row(
    jwt_client, monkeypatch, fake_search, test_model
):
    import app.agents.assistant as assistant_module

    monkeypatch.setattr(assistant_module.document_tools, "search_documents", fake_search)

    tenant_id = uuid.uuid4()
    identity_id = uuid.uuid4()
    _install_fake_control_plane(
        monkeypatch,
        auth_settings={tenant_id: (ISSUER, False)},
        identities={(ISSUER, "sub-1"): identity_id},
        memberships={(tenant_id, identity_id): "support"},
    )

    captured_contexts: list[RequestContext] = []
    original_run_assistant = assistant_module.run_assistant

    async def _capturing_run_assistant(prompt, deps):
        captured_contexts.append(deps.ctx)
        return await original_run_assistant(prompt, deps)

    monkeypatch.setattr("app.api.agents.run_assistant", _capturing_run_assistant)

    token = _make_token(audience=str(tenant_id))
    response = await _post_run(tenant_id, token)
    assert response.status_code == 200, response.text
    assert captured_contexts, "run_assistant should have been called with a context"
    ctx = captured_contexts[0]
    assert ctx.tenant_id == tenant_id
    assert ctx.identity_id == identity_id
    assert ctx.roles == frozenset({"support"})
