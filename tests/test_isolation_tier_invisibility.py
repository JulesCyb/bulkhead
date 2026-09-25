"""S10-T6 (#78): a tenant's own request can never observe or influence its isolation tier or
database alias (ADR-0002, ADR-0011). Two ASGI-application-seam checks:

- no response schema anywhere in the API exposes a database alias or isolation-tier field;
- the tenant-editable settings model (`app.tenant_settings.TenantSettings`) rejects a
  database-alias-shaped or DSN-shaped key, and `Settings` (process configuration) exposes no
  field capable of holding a second tenant's connection string.

No DB, no real model call.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.main import app
from app.tenant_settings import TenantSettings

# Substrings that would mark an OpenAPI schema property, or a Settings field, as capable of
# naming/holding an isolation tier or database alias — matched against the lowercased,
# underscore-normalized name, mirroring app.tenant_settings._normalize_key.
_FORBIDDEN_MARKERS = ("database_alias", "db_alias", "isolation_tier")


def _normalize(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _walk_schema_properties(node: object) -> list[str]:
    """Every property name appearing anywhere in an OpenAPI (JSON Schema) document, including
    nested $defs/components and array items — not just the top level of each response model."""
    names: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                names.extend(value.keys())
            names.extend(_walk_schema_properties(value))
    elif isinstance(node, list):
        for item in node:
            names.extend(_walk_schema_properties(item))
    return names


def test_no_response_schema_exposes_isolation_tier_or_database_alias():
    schema = app.openapi()
    property_names = _walk_schema_properties(schema.get("components", {}).get("schemas", {}))
    # Also walk every path's response bodies directly, in case a route declares an inline schema
    # rather than a named component.
    property_names += _walk_schema_properties(schema.get("paths", {}))

    offending = [name for name in property_names if _normalize(name) in _FORBIDDEN_MARKERS]
    assert offending == [], f"response schema exposes forbidden field(s): {offending}"


def test_tenant_settings_rejects_database_alias_shaped_key():
    with pytest.raises(ValidationError):
        TenantSettings.model_validate({"database_alias": "eu1"})
    with pytest.raises(ValidationError):
        TenantSettings.model_validate({"db-alias": "eu1"})
    with pytest.raises(ValidationError):
        TenantSettings.model_validate({"isolation_tier": "dedicated"})


def test_tenant_settings_rejects_dsn_shaped_value_in_any_field():
    with pytest.raises(ValidationError):
        TenantSettings.model_validate(
            {"model": "postgresql+asyncpg://app:app@dedicated-host:5432/tenant42"}
        )
    with pytest.raises(ValidationError):
        TenantSettings.model_validate({"connection_string": "postgres://u:p@host/db"})


def test_tenant_settings_rejects_unknown_keys_outright():
    """extra="forbid" on top of the marker/DSN checks: a tenant may only ever set the fields
    this model declares (today: model)."""
    with pytest.raises(ValidationError):
        TenantSettings.model_validate({"some_new_field": "value"})


def test_tenant_settings_accepts_known_tenant_preferences():
    settings = TenantSettings.model_validate({"model": "anthropic:claude-sonnet-4-5"})
    assert settings.model == "anthropic:claude-sonnet-4-5"


def test_tenant_settings_rejects_residency():
    """Residency is an operator-owned control-plane fact (ADR-0008), never a tenant setting."""
    with pytest.raises(ValidationError):
        TenantSettings.model_validate({"residency": "us"})


def test_settings_has_no_field_capable_of_holding_a_second_tenants_connection_string():
    """`Settings.database_url` is this deployment's own single pooled-database connection
    string (app/db/session.py); a dedicated tenant's DSN lives only in a tenant-secret file
    keyed by alias (app/db/engine_registry.py), never as a field here. Guards against a future
    field such as `dedicated_database_url`/`tenant_database_url` being added directly to
    process configuration instead of resolved per-alias from the secrets directory."""
    field_names = set(Settings.model_fields)
    suspicious = {
        name
        for name in field_names
        if _normalize(name) in _FORBIDDEN_MARKERS
        or ("database" in _normalize(name) and "url" in _normalize(name) and name != "database_url")
    }
    assert suspicious == set(), f"Settings exposes a suspicious field: {suspicious}"
