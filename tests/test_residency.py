"""Residency route resolution (Spec 8 / #60, ADR-0008): the resolver logic exercised without a
real database -- the RLS-filtered read itself is covered by the embedded-Postgres tests in
tests/test_rls_integration.py. Mirrors tests/test_gateway_credentials.py's fake-session seam.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import RESIDENCY_ALLOW_LIST, Settings
from app.context import RequestContext
from app.gateway_credentials import GatewayCredentialUnavailable
from app.residency import ResidencyUnresolved, resolve_residency_route


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_url="postgresql+asyncpg://app:app@localhost:5432/app",
        gateway_credentials_dir=str(tmp_path),
    )


def _fake_session(residency_row: tuple[str] | None, alias_row: tuple[str] | None) -> AsyncMock:
    """A session whose `execute(...).first()` returns, in order, the residency read then the
    gateway-credential-alias read -- `.first()` is sync on a real SQLAlchemy `Result`, so a plain
    `MagicMock` per call, not an `AsyncMock` (whose attributes are themselves awaitable)."""
    residency_result = MagicMock()
    residency_result.first.return_value = residency_row
    alias_result = MagicMock()
    alias_result.first.return_value = alias_row

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[residency_result, alias_result])
    return session


async def test_unset_residency_raises_typed_error() -> None:
    ctx = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())
    session = _fake_session(None, None)
    with pytest.raises(ResidencyUnresolved):
        await resolve_residency_route(session, ctx)


async def test_null_residency_raises_typed_error() -> None:
    ctx = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())
    session = _fake_session((None,), None)
    with pytest.raises(ResidencyUnresolved):
        await resolve_residency_route(session, ctx)


async def test_unknown_residency_raises_typed_error_not_a_fallback() -> None:
    ctx = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())
    session = _fake_session(("mars",), None)
    with pytest.raises(ResidencyUnresolved):
        await resolve_residency_route(session, ctx)


async def test_resolved_route_matches_the_allow_list_entry(tmp_path, settings) -> None:
    (tmp_path / "acme-gateway-key").write_text("sk-acme-secret")
    ctx = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())
    session = _fake_session(("eu",), ("acme-gateway-key",))

    resolved = await resolve_residency_route(session, ctx, settings=settings)

    assert resolved.residency == "eu"
    assert resolved.route == RESIDENCY_ALLOW_LIST["eu"]
    assert resolved.gateway_credential.get_secret_value() == "sk-acme-secret"


async def test_missing_gateway_credential_propagates_unchanged(settings) -> None:
    """The residency itself resolves fine; the credential lookup below it fails closed with its
    own typed error rather than being masked by a residency-shaped one."""
    ctx = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())
    session = _fake_session(("eu",), None)
    with pytest.raises(GatewayCredentialUnavailable):
        await resolve_residency_route(session, ctx, settings=settings)


async def test_two_tenants_different_residencies_never_cross(tmp_path, settings) -> None:
    (tmp_path / "acme-gateway-key").write_text("sk-acme-secret")
    (tmp_path / "globex-gateway-key").write_text("sk-globex-secret")

    ctx_eu = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())
    session_eu = _fake_session(("eu",), ("acme-gateway-key",))
    ctx_us = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())
    session_us = _fake_session(("us",), ("globex-gateway-key",))

    resolved_eu = await resolve_residency_route(session_eu, ctx_eu, settings=settings)
    resolved_us = await resolve_residency_route(session_us, ctx_us, settings=settings)

    assert resolved_eu.residency == "eu"
    assert resolved_us.residency == "us"
    assert resolved_eu.route != resolved_us.route
    assert resolved_eu.route.trace_sink_host != resolved_us.route.trace_sink_host
    assert (
        resolved_eu.gateway_credential.get_secret_value()
        != resolved_us.gateway_credential.get_secret_value()
    )


def test_resolved_route_is_frozen() -> None:
    """The composed route is immutable -- nothing downstream can splice in a different
    residency's route or another tenant's credential after the fact."""
    from pydantic import ValidationError

    resolved = _build_resolved_for_frozen_check()
    with pytest.raises(ValidationError):
        resolved.residency = "us"  # type: ignore[misc]


def _build_resolved_for_frozen_check():
    from pydantic import SecretStr

    from app.residency import ResolvedResidencyRoute

    return ResolvedResidencyRoute(
        residency="eu",
        route=RESIDENCY_ALLOW_LIST["eu"],
        gateway_credential=SecretStr("sk-test"),
    )
