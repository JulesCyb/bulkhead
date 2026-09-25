"""Per-tenant gateway credential resolution (Spec 7 / #52): the file-reading half and the typed
error, exercised without a real database -- the control-plane lookup itself is covered by the
embedded-Postgres tests in tests/test_rls_integration.py. `settings` fixture reuses the same
`Settings` object the real app constructs; only `gateway_credentials_dir` is pointed at a tmp
directory, so this is the "ASGI application plus Settings" seam: no DB, no HTTP, real Settings.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import Settings
from app.context import RequestContext
from app.gateway_credentials import (
    GatewayCredentialUnavailable,
    read_gateway_credential,
    resolve_gateway_credential,
    resolve_gateway_credential_alias,
)


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


def _fake_session(row: tuple[str] | None) -> AsyncMock:
    """A session whose `execute(...).first()` returns `row` -- `.first()` is sync on a real
    SQLAlchemy `Result`, so the returned object is a plain `MagicMock`, not an `AsyncMock`
    (an `AsyncMock`'s attributes are themselves awaitable by default, which `.first()` is not).
    """
    result = MagicMock()
    result.first.return_value = row
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    return session


async def test_alias_resolution_raises_typed_error_when_none_recorded():
    ctx = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())
    session = _fake_session(None)
    with pytest.raises(GatewayCredentialUnavailable):
        await resolve_gateway_credential_alias(session, ctx)


async def test_end_to_end_resolver_reads_the_aliased_file(tmp_path, settings):
    ctx = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())
    session = _fake_session(("acme-gateway-key",))
    (tmp_path / "acme-gateway-key").write_text("sk-acme-secret")

    credential = await resolve_gateway_credential(session, ctx, settings=settings)
    assert credential.get_secret_value() == "sk-acme-secret"


async def test_end_to_end_resolver_two_tenants_two_aliases_two_different_credentials(
    tmp_path, settings
):
    (tmp_path / "acme-gateway-key").write_text("sk-acme-secret")
    (tmp_path / "globex-gateway-key").write_text("sk-globex-secret")

    ctx_acme = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())
    session_acme = _fake_session(("acme-gateway-key",))

    ctx_globex = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())
    session_globex = _fake_session(("globex-gateway-key",))

    acme_credential = await resolve_gateway_credential(session_acme, ctx_acme, settings=settings)
    globex_credential = await resolve_gateway_credential(
        session_globex, ctx_globex, settings=settings
    )
    assert acme_credential.get_secret_value() == "sk-acme-secret"
    assert globex_credential.get_secret_value() == "sk-globex-secret"
    assert acme_credential.get_secret_value() != globex_credential.get_secret_value()
