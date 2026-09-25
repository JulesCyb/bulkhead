"""Settings-level test (#24): a tenant's per-request issuer falls back to the process-wide
default when its own configured issuer is unset, and keeps its own when it has one -- exercising
`TenantAuthSettingsRepository.get`'s own fallback logic directly against a fake session (no
database; see tests/test_rls_integration.py for the real-Postgres counterpart of this same
behaviour, `test_tenant_auth_settings_falls_back_to_default_and_reports_suspension`)."""

from __future__ import annotations

import uuid

from app.repositories.control import TenantAuthSettingsRepository


class _FakeResult:
    def __init__(self, row: dict | None) -> None:
        self._row = row

    def mappings(self) -> _FakeResult:
        return self

    def one_or_none(self) -> dict | None:
        return self._row


class _FakeSession:
    def __init__(self, row: dict | None) -> None:
        self._row = row

    async def execute(self, *args, **kwargs) -> _FakeResult:
        return _FakeResult(self._row)


async def test_tenant_with_no_configured_issuer_falls_back_to_the_default():
    session = _FakeSession({"identity_issuer": None, "suspended_at": None})
    settings = await TenantAuthSettingsRepository().get(
        session, tenant_id=uuid.uuid4(), default_issuer="https://default.example.com"
    )
    assert settings is not None
    assert settings.issuer == "https://default.example.com"
    assert settings.suspended is False


async def test_tenant_with_its_own_issuer_keeps_it_over_the_default():
    session = _FakeSession(
        {"identity_issuer": "https://tenant-own-idp.example.com", "suspended_at": None}
    )
    settings = await TenantAuthSettingsRepository().get(
        session, tenant_id=uuid.uuid4(), default_issuer="https://default.example.com"
    )
    assert settings is not None
    assert settings.issuer == "https://tenant-own-idp.example.com"
    assert settings.suspended is False


async def test_no_control_plane_row_returns_none_rather_than_a_default():
    """No row for the tenant at all (the ordinary case for a pooled tenant with no override,
    ADR-0002) is distinct from "issuer unset" -- callers (app/deps.py) apply the default
    themselves in that case rather than this repository inventing one."""
    session = _FakeSession(None)
    settings = await TenantAuthSettingsRepository().get(
        session, tenant_id=uuid.uuid4(), default_issuer="https://default.example.com"
    )
    assert settings is None
