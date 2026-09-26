"""Residency route resolution (Spec 8 / #60, ADR-0008; #105): the resolver is a function of the
tenant record (`app.tenant_record.TenantRecord`) and settings -- every case below constructs the
record directly, no database and no fake session. The record read itself (RLS-scoped) is covered
by the embedded-Postgres tests in tests/test_rls_integration.py.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import pytest

from app.config import Settings
from app.gateway_credentials import GatewayCredentialUnavailable
from app.residency import ResidencyAllowList, ResidencyUnresolved, resolve_residency_route
from app.tenant_record import TenantRecord

ALLOW_LIST = ResidencyAllowList.load()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_url="postgresql+asyncpg://app:app@localhost:5432/app",
        gateway_credentials_dir=str(tmp_path),
    )


def _record(residency: str | None, alias: str | None = None) -> TenantRecord:
    return TenantRecord(tenant_id=uuid.uuid4(), residency=residency, gateway_credential_alias=alias)


def test_unset_residency_raises_typed_error(settings) -> None:
    with pytest.raises(ResidencyUnresolved):
        resolve_residency_route(_record(None, "acme-gateway-key"), settings=settings)


def test_empty_residency_raises_typed_error(settings) -> None:
    with pytest.raises(ResidencyUnresolved):
        resolve_residency_route(_record("", "acme-gateway-key"), settings=settings)


def test_unknown_residency_raises_typed_error_not_a_fallback(settings) -> None:
    with pytest.raises(ResidencyUnresolved):
        resolve_residency_route(_record("mars", "acme-gateway-key"), settings=settings)


def test_unresolved_residency_is_refused_before_the_credential_is_read(tmp_path, settings) -> None:
    """Fail closed on the residency first: a tenant with no usable residency never has its
    credential file read at all -- no alias recorded here, so a credential read would raise the
    other typed error instead."""
    with pytest.raises(ResidencyUnresolved):
        resolve_residency_route(_record(None, None), settings=settings)


def test_resolved_route_matches_the_allow_list_entry(tmp_path, settings) -> None:
    (tmp_path / "acme-gateway-key").write_text("sk-acme-secret")

    resolved = resolve_residency_route(_record("eu", "acme-gateway-key"), settings=settings)

    assert resolved.residency == "eu"
    assert resolved.route == ALLOW_LIST.route_for("eu")
    assert resolved.gateway_credential.get_secret_value() == "sk-acme-secret"


def test_missing_gateway_credential_alias_fails_closed(settings) -> None:
    """The residency itself resolves fine; a record with no credential alias fails closed with
    the credential's own typed error rather than being masked by a residency-shaped one."""
    with pytest.raises(GatewayCredentialUnavailable):
        resolve_residency_route(_record("eu", None), settings=settings)


def test_missing_gateway_credential_file_fails_closed(settings) -> None:
    with pytest.raises(GatewayCredentialUnavailable):
        resolve_residency_route(_record("eu", "no-such-alias"), settings=settings)


def test_two_tenants_different_residencies_never_cross(tmp_path, settings) -> None:
    (tmp_path / "acme-gateway-key").write_text("sk-acme-secret")
    (tmp_path / "globex-gateway-key").write_text("sk-globex-secret")

    resolved_eu = resolve_residency_route(_record("eu", "acme-gateway-key"), settings=settings)
    resolved_us = resolve_residency_route(_record("us", "globex-gateway-key"), settings=settings)

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
        route=ALLOW_LIST.route_for("eu"),
        gateway_credential=SecretStr("sk-test"),
    )


# --- The tenant record is the only read of residency and settings (#105) -----------------------

_APP = Path(__file__).resolve().parent.parent / "app"

# Each of these was once a per-request control-plane or settings read; model, embedding, and
# tracing resolution are functions of the tenant record now, and the record's own read
# (`ControlRepository.get_tenant_record`) reads the view and the settings row itself.
_RETIRED_READS = {
    "residency read": re.compile(r"\bget_residency\("),
    "gateway credential alias read": re.compile(r"\bget_gateway_credential_alias\("),
    "per-context settings read": re.compile(r"TenantSettingsRepository\(\)\.get\("),
    "session-taking tracing selection": re.compile(r"\bresolve_tenant_tracing_selection\b"),
}


def test_residency_and_settings_are_read_from_the_record_only() -> None:
    found = {
        (name, path.relative_to(_APP.parent).as_posix())
        for path in _APP.rglob("*.py")
        for name, pattern in _RETIRED_READS.items()
        if pattern.search(path.read_text())
    }
    assert found == set()
