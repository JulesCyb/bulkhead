"""Unit seams for #84 (cap tenant-configurable conversation retention, ADR-0006; GDPR
Art. 5(1)(e) storage limitation): no DB, no real model call.

- `Settings.max_retention_days` refuses construction below 1 day, and below the documented
  `DEFAULT_RETENTION_DAYS` (90) -- a maximum under the default would put every tenant that has
  never set its own `retention_days` over the cap by definition.
- `TenantSettings.require_retention_within_cap` -- the write-side half -- rejects a value above
  the cap it is handed and passes an unset one or one at/below the cap.
- `app.tenant_settings.effective_retention_days` -- the one function the retention job calls --
  returns the stored value unchanged when at or below the cap, the cap when the stored value is
  above it, and `DEFAULT_RETENTION_DAYS` when the tenant never set one at all.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.tenant_settings import (
    DEFAULT_RETENTION_DAYS,
    RetentionDaysExceedsMaximumError,
    TenantSettings,
    effective_retention_days,
)

_VALID_KWARGS = {"embedding_provider": "openai", "embedding_model": "text-embedding-3-small"}


def test_settings_max_retention_days_has_a_documented_default(monkeypatch):
    monkeypatch.delenv("MAX_RETENTION_DAYS", raising=False)
    settings = Settings(**_VALID_KWARGS)
    assert settings.max_retention_days == 365


def test_settings_max_retention_days_overridable_via_env(monkeypatch):
    monkeypatch.setenv("MAX_RETENTION_DAYS", "180")
    settings = Settings(**_VALID_KWARGS)
    assert settings.max_retention_days == 180


def test_settings_rejects_max_retention_days_below_one():
    with pytest.raises(ValidationError):
        Settings(**_VALID_KWARGS, max_retention_days=0)
    with pytest.raises(ValidationError):
        Settings(**_VALID_KWARGS, max_retention_days=-5)


def test_settings_rejects_max_retention_days_below_the_default():
    """A deployment maximum below DEFAULT_RETENTION_DAYS would make the documented default itself
    exceed the cap for every tenant that has never set its own value -- refused outright rather
    than silently reconciled by the read-side clamp."""
    with pytest.raises(ValidationError):
        Settings(**_VALID_KWARGS, max_retention_days=DEFAULT_RETENTION_DAYS - 1)


def test_settings_accepts_max_retention_days_equal_to_the_default():
    settings = Settings(**_VALID_KWARGS, max_retention_days=DEFAULT_RETENTION_DAYS)
    assert settings.max_retention_days == DEFAULT_RETENTION_DAYS


def test_effective_retention_days_returns_the_stored_value_when_at_or_below_the_cap():
    settings = TenantSettings(retention_days=30)
    assert effective_retention_days(settings, max_days=365) == 30
    # Exactly at the cap: honored as-is, not clamped down further.
    at_cap = TenantSettings(retention_days=365)
    assert effective_retention_days(at_cap, max_days=365) == 365


def test_effective_retention_days_clamps_a_stored_value_above_the_cap():
    settings = TenantSettings(retention_days=10_000)
    assert effective_retention_days(settings, max_days=365) == 365


def test_effective_retention_days_uses_the_default_when_unset():
    settings = TenantSettings()
    assert settings.retention_days is None
    assert effective_retention_days(settings, max_days=365) == DEFAULT_RETENTION_DAYS


def test_tenant_settings_rejects_a_retention_days_above_the_maximum():
    with pytest.raises(RetentionDaysExceedsMaximumError, match="exceeds this deployment's maximum"):
        TenantSettings(retention_days=366).require_retention_within_cap(max_days=365)


def test_tenant_settings_accepts_retention_days_at_or_below_the_maximum_or_unset():
    TenantSettings(retention_days=365).require_retention_within_cap(max_days=365)
    TenantSettings(retention_days=1).require_retention_within_cap(max_days=365)
    TenantSettings().require_retention_within_cap(max_days=365)
