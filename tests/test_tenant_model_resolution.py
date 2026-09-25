"""Per-tenant model/embedding resolution (Spec 7 / #54, ADR-0009): the allow-list check, the
per-tenant client cache, and the wall-clock deadline -- exercised at the "ASGI application plus
Settings" seam: real `Settings`, a fake control-plane session (same style as
tests/test_gateway_credentials.py), no real database and no real model/provider call.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import APITimeoutError, AsyncOpenAI
from pydantic import SecretStr
from pydantic_ai.models.openai import OpenAIChatModel

from app.config import Settings
from app.context import RequestContext
from app.embeddings import (
    build_tenant_embedding_client,
    reset_tenant_embedding_client_cache,
    resolve_tenant_embedding_client,
)
from app.gateway_credentials import GatewayCredentialUnavailable
from app.llm import (
    ModelNotAllowedForResidency,
    build_tenant_chat_model,
    reset_tenant_chat_model_cache,
    resolve_tenant_chat_model,
    validate_model_for_residency,
)


@pytest.fixture(autouse=True)
def _reset_caches():
    reset_tenant_chat_model_cache()
    reset_tenant_embedding_client_cache()
    yield
    reset_tenant_chat_model_cache()
    reset_tenant_embedding_client_cache()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_url="postgresql+asyncpg://app:app@localhost:5432/app",
        gateway_credentials_dir=str(tmp_path),
        litellm_base_url="http://litellm.internal:4000",
    )


def _fake_session(alias_row, residency_row) -> AsyncMock:
    """A session whose `execute(...).first()` returns whichever row matches the query it was
    called with, in the order `resolve_tenant_chat_model` issues them (residency, then alias).
    Mirrors `tests/test_gateway_credentials.py`'s `_fake_session` helper."""
    rows = iter([residency_row, alias_row])

    async def _execute(*_args, **_kwargs):
        result = MagicMock()
        result.first.return_value = next(rows)
        return result

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=_execute)
    return session


# --- Model allow-list validation (AC1) -------------------------------------------------------


def test_model_on_the_allow_list_for_its_residency_is_accepted():
    assert validate_model_for_residency("claude-eu", "eu") == "claude-eu"
    assert validate_model_for_residency("openai:claude", "us") == "claude"


def test_model_outside_the_allow_list_for_its_residency_is_rejected():
    with pytest.raises(ModelNotAllowedForResidency):
        validate_model_for_residency("claude", "eu")  # "claude" is the US alias, not EU's


def test_model_valid_for_a_different_residency_is_still_rejected_for_this_one():
    with pytest.raises(ModelNotAllowedForResidency):
        validate_model_for_residency("claude-eu", "us")


def test_unknown_residency_has_no_allowed_models_at_all():
    with pytest.raises(ModelNotAllowedForResidency):
        validate_model_for_residency("claude", "atlantis")


async def test_rejected_model_never_reaches_credential_resolution_or_client_construction(
    settings, monkeypatch
):
    """AC1: rejected before any client is constructed and before any network call -- the
    credential resolver must never even be called."""
    from app import llm as llm_module

    called = False

    async def _fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("credential resolution must not run for a rejected model")

    monkeypatch.setattr(llm_module, "resolve_gateway_credential", _fail_if_called)

    ctx = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())
    session = _fake_session(alias_row=None, residency_row=("eu",))

    with pytest.raises(ModelNotAllowedForResidency):
        await resolve_tenant_chat_model(session, ctx, "claude", settings=settings)
    assert called is False


# --- End-to-end resolution against a fake control-plane session (AC2) -----------------------


async def test_model_on_the_list_resolves_using_the_tenants_own_credential(settings, tmp_path):
    (tmp_path / "acme-gateway-key").write_text("sk-acme-secret")
    ctx = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())
    session = _fake_session(alias_row=("acme-gateway-key",), residency_row=("eu",))

    model = await resolve_tenant_chat_model(session, ctx, "claude-eu", settings=settings)

    assert isinstance(model, OpenAIChatModel)
    assert model.provider.client.api_key == "sk-acme-secret"


async def test_missing_gateway_credential_raises_the_prior_tickets_typed_error(settings):
    """The credential resolver's own typed error (#52) surfaces unchanged -- resolution here
    doesn't mask it behind a different exception."""
    ctx = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())
    session = _fake_session(alias_row=None, residency_row=("eu",))

    with pytest.raises(GatewayCredentialUnavailable):
        await resolve_tenant_chat_model(session, ctx, "claude-eu", settings=settings)


async def test_missing_control_plane_residency_fails_closed(settings):
    """A tenant with no recorded residency never inherits the deployment's (ADR-0008)."""
    from app.residency import ResidencyUnresolved

    ctx = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())
    session = _fake_session(alias_row=None, residency_row=None)

    with pytest.raises(ResidencyUnresolved):
        await resolve_tenant_chat_model(session, ctx, "claude-eu", settings=settings)


# --- Per-tenant client caching (AC3) ----------------------------------------------------------


def test_two_different_tenants_get_two_distinct_chat_clients(settings):
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    model_a = build_tenant_chat_model(
        tenant_a, "claude-eu", SecretStr("secret-a"), settings=settings
    )
    model_b = build_tenant_chat_model(
        tenant_b, "claude-eu", SecretStr("secret-b"), settings=settings
    )
    assert model_a is not model_b
    assert model_a.provider.client is not model_b.provider.client


def test_the_same_tenant_twice_reuses_one_chat_client(settings):
    tenant_id = uuid.uuid4()
    first = build_tenant_chat_model(tenant_id, "claude-eu", SecretStr("secret"), settings=settings)
    second = build_tenant_chat_model(
        tenant_id, "claude-eu", SecretStr("a-different-secret"), settings=settings
    )
    assert first is second


def test_two_different_tenants_get_two_distinct_embedding_clients(settings):
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    client_a = build_tenant_embedding_client(tenant_a, SecretStr("secret-a"), settings=settings)
    client_b = build_tenant_embedding_client(tenant_b, SecretStr("secret-b"), settings=settings)
    assert client_a is not client_b


def test_the_same_tenant_twice_reuses_one_embedding_client(settings):
    tenant_id = uuid.uuid4()
    first = build_tenant_embedding_client(tenant_id, SecretStr("secret"), settings=settings)
    second = build_tenant_embedding_client(
        tenant_id, SecretStr("a-different-secret"), settings=settings
    )
    assert first is second


def test_building_a_chat_client_with_no_gateway_configured_refuses(tmp_path):
    settings_without_gateway = Settings(
        database_url="postgresql+asyncpg://app:app@localhost:5432/app",
        gateway_credentials_dir=str(tmp_path),
    )
    with pytest.raises(RuntimeError):
        build_tenant_chat_model(
            uuid.uuid4(), "claude-eu", SecretStr("secret"), settings=settings_without_gateway
        )


# --- Wall-clock deadlines (AC4) ----------------------------------------------------------------


def test_chat_client_carries_the_configured_call_deadline(settings):
    model = build_tenant_chat_model(
        uuid.uuid4(), "claude-eu", SecretStr("secret"), settings=settings
    )
    assert model.settings is not None
    assert model.settings["timeout"] == settings.llm_call_timeout_seconds


async def test_an_engineered_embedding_call_that_stalls_past_the_deadline_ends(settings):
    """A real behavioral test of the deadline, with no real provider: a local TCP server that
    accepts the connection and then never answers must not be allowed to hang the call past the
    configured deadline. `httpx.MockTransport` can't stand in here -- it bypasses the socket
    layer httpx's own timeout enforcement operates at, so a stalling *handler* would never
    actually be interrupted by the timeout being tested.
    """

    async def _never_respond(_reader: asyncio.StreamReader, _writer: asyncio.StreamWriter) -> None:
        # Deliberately short (not the 30s+ a real stalled provider might take): a longer sleep
        # here would keep this connection open past the test itself, and `server.close()` below
        # (unlike `async with server:`) does not wait for open connections to finish.
        await asyncio.sleep(2)

    server = await asyncio.start_server(_never_respond, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    try:
        settings_short_deadline = settings.model_copy(
            update={
                "litellm_base_url": f"http://{host}:{port}",
                "embedding_call_timeout_seconds": 0.2,
            }
        )
        client = build_tenant_embedding_client(
            uuid.uuid4(), SecretStr("secret"), settings=settings_short_deadline
        )

        start = asyncio.get_event_loop().time()
        with pytest.raises(APITimeoutError):
            await asyncio.wait_for(
                client.embeddings.create(model="embeddings", input="hi"), timeout=5
            )
        assert asyncio.get_event_loop().time() - start < 5
    finally:
        server.close()


def test_embedding_client_carries_the_configured_call_deadline(settings):
    client: AsyncOpenAI = build_tenant_embedding_client(
        uuid.uuid4(), SecretStr("secret"), settings=settings
    )
    assert client.timeout == settings.embedding_call_timeout_seconds


async def test_resolve_tenant_embedding_client_uses_the_tenants_own_credential(settings, tmp_path):
    (tmp_path / "acme-gateway-key").write_text("sk-acme-secret")
    ctx = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())
    # resolve_tenant_embedding_client now routes through resolve_residency_route (#61), which
    # reads residency first, then the gateway-credential alias -- same order as the chat path.
    session = _fake_session(alias_row=("acme-gateway-key",), residency_row=("eu",))

    client = await resolve_tenant_embedding_client(session, ctx, settings=settings)
    assert client.api_key == "sk-acme-secret"


async def test_resolve_tenant_embedding_client_fails_closed_with_no_resolvable_residency(settings):
    """#61: the embedding path fails closed on an unresolvable residency exactly like the chat
    path does -- it must never fall back to a default embedding endpoint."""
    from app.residency import ResidencyUnresolved

    ctx = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())
    session = _fake_session(alias_row=None, residency_row=None)

    with pytest.raises(ResidencyUnresolved):
        await resolve_tenant_embedding_client(session, ctx, settings=settings)
