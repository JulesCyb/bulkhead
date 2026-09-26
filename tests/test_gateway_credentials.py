"""Per-tenant gateway credential resolution (Spec 7 / #52, #105): the file-reading half and the
typed error, exercised without a real database -- the alias comes from the tenant record
(`app.tenant_record.TenantRecord`, constructed directly here), whose read is covered by the
embedded-Postgres tests in tests/test_rls_integration.py. `settings` fixture reuses the same
`Settings` object the real app constructs; only `gateway_credentials_dir` is pointed at a tmp
directory, so this is the "ASGI application plus Settings" seam: no DB, no HTTP, real Settings.
"""

from __future__ import annotations

import uuid

import pytest

from app.config import Settings
from app.gateway_credentials import (
    GatewayCredentialUnavailable,
    read_gateway_credential,
    resolve_gateway_credential,
)
from app.tenant_record import TenantRecord


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_url="postgresql+asyncpg://app:app@localhost:5432/app",
        gateway_credentials_dir=str(tmp_path),
    )


def test_reads_the_file_named_by_the_alias(tmp_path, settings):
    (tmp_path / "acme-gateway-key").write_text("sk-acme-secret\n")
    credential = read_gateway_credential("acme-gateway-key", settings=settings)
    assert credential.get_secret_value() == "sk-acme-secret"


def test_two_aliases_read_two_different_files(tmp_path, settings):
    (tmp_path / "acme").write_text("secret-a")
    (tmp_path / "globex").write_text("secret-b")
    a = read_gateway_credential("acme", settings=settings)
    b = read_gateway_credential("globex", settings=settings)
    assert a.get_secret_value() == "secret-a"
    assert b.get_secret_value() == "secret-b"


def test_missing_secret_file_raises_typed_error_not_raw_io_error(settings):
    with pytest.raises(GatewayCredentialUnavailable):
        read_gateway_credential("no-such-alias", settings=settings)


def test_empty_secret_file_raises_typed_error(tmp_path, settings):
    (tmp_path / "blank").write_text("")
    with pytest.raises(GatewayCredentialUnavailable):
        read_gateway_credential("blank", settings=settings)


def test_nothing_is_cached_across_calls_a_rotated_file_is_picked_up(tmp_path, settings):
    path = tmp_path / "acme"
    path.write_text("first-value")
    first = read_gateway_credential("acme", settings=settings)
    path.write_text("rotated-value")
    second = read_gateway_credential("acme", settings=settings)
    assert first.get_secret_value() == "first-value"
    assert second.get_secret_value() == "rotated-value"


def _record(alias: str | None) -> TenantRecord:
    return TenantRecord(tenant_id=uuid.uuid4(), residency="eu", gateway_credential_alias=alias)


@pytest.mark.parametrize("alias", [None, ""])
def test_a_record_without_an_alias_raises_typed_error(settings, alias):
    with pytest.raises(GatewayCredentialUnavailable):
        resolve_gateway_credential(_record(alias), settings=settings)


def test_end_to_end_resolver_reads_the_aliased_file(tmp_path, settings):
    (tmp_path / "acme-gateway-key").write_text("sk-acme-secret")

    credential = resolve_gateway_credential(_record("acme-gateway-key"), settings=settings)
    assert credential.get_secret_value() == "sk-acme-secret"


def test_end_to_end_resolver_two_tenants_two_aliases_two_different_credentials(tmp_path, settings):
    (tmp_path / "acme-gateway-key").write_text("sk-acme-secret")
    (tmp_path / "globex-gateway-key").write_text("sk-globex-secret")

    acme_credential = resolve_gateway_credential(_record("acme-gateway-key"), settings=settings)
    globex_credential = resolve_gateway_credential(_record("globex-gateway-key"), settings=settings)
    assert acme_credential.get_secret_value() == "sk-acme-secret"
    assert globex_credential.get_secret_value() == "sk-globex-secret"
