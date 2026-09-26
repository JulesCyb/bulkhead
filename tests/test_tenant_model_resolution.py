"""Per-tenant model/embedding resolution (Spec 7 / #54, ADR-0009, #105): the allow-list check,
the tenant's own `model` setting, the per-tenant client cache, and the wall-clock deadline.

The resolvers are functions of the tenant record (`app.tenant_record.TenantRecord`) and
`Settings` (#105): every case constructs the record directly -- no database, no fake session, no
real model/provider call.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from openai import APITimeoutError, AsyncOpenAI
from pydantic import SecretStr
from pydantic_ai.models.openai import OpenAIChatModel

from app.config import Settings
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
from app.residency import ResidencyAllowList, ResidencyUnresolved
from app.tenant_record import TenantRecord
from app.tenant_settings import TenantSettings


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


# `eu` allows two chat models here (the shipped config/residency.toml allows only one), so a
# tenant's own choice is distinguishable from the deployment default (`claude-eu`).
_TWO_EU_MODELS = ResidencyAllowList.from_data(
    {
        "residency": {
            "eu": {
                "model_host_patterns": ["litellm", "litellm.internal"],
                "embedding_endpoint": "https://gateway-eu.internal/v1",
                "trace_sink_host": "eu.cloud.langfuse.com",
                "models": ["claude-eu", "claude-eu-large", "embeddings"],
            },
            "us": {
                "model_host_patterns": ["gateway-us.internal"],
                "embedding_endpoint": "https://gateway-us.internal/v1",
                "trace_sink_host": "us.cloud.langfuse.com",
                "models": ["claude", "embeddings"],
            },
        }
    }
)


@pytest.fixture
def two_model_settings(settings) -> Settings:
    settings.residency_allow_list = _TWO_EU_MODELS
    return settings


def _record(
    residency: str | None = "eu",
    alias: str | None = "acme-gateway-key",
    model: str | None = None,
    tenant_id: uuid.UUID | None = None,
) -> TenantRecord:
    return TenantRecord(
        tenant_id=tenant_id or uuid.uuid4(),
        residency=residency,
        gateway_credential_alias=alias,
        settings=TenantSettings(model=model),
    )


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


def test_rejected_model_never_reaches_credential_resolution_or_client_construction(
    settings, monkeypatch
):
    """AC1: rejected before any client is constructed and before any network call -- the
    credential resolver must never even be called (ADR-0009's order)."""
    from app import llm as llm_module

    called = False

    def _fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("credential resolution must not run for a rejected model")

    monkeypatch.setattr(llm_module, "resolve_gateway_credential", _fail_if_called)

    with pytest.raises(ModelNotAllowedForResidency):
        resolve_tenant_chat_model(_record(model="claude"), settings=settings)
    assert called is False


# --- End to end from the record (AC2, #105) ---------------------------------------------------


def test_model_on_the_list_resolves_using_the_tenants_own_credential(settings, tmp_path):
    (tmp_path / "acme-gateway-key").write_text("sk-acme-secret")

    model = resolve_tenant_chat_model(_record(model="claude-eu"), settings=settings)

    assert isinstance(model, OpenAIChatModel)
    assert model.provider.client.api_key == "sk-acme-secret"


def test_a_tenant_without_a_model_setting_runs_the_deployment_default(two_model_settings, tmp_path):
    (tmp_path / "acme-gateway-key").write_text("sk-acme-secret")

    model = resolve_tenant_chat_model(_record(model=None), settings=two_model_settings)

    assert model.model_name == "claude-eu"  # Settings.llm_model


def test_the_tenants_own_allow_listed_model_setting_is_the_model_it_runs(
    two_model_settings, tmp_path
):
    """#105: `tenants.settings["model"]` is live -- the record's setting wins over the
    deployment default when it is on the residency's allow-list."""
    (tmp_path / "acme-gateway-key").write_text("sk-acme-secret")

    model = resolve_tenant_chat_model(_record(model="claude-eu-large"), settings=two_model_settings)

    assert model.model_name == "claude-eu-large"


def test_a_provider_prefixed_model_setting_resolves_to_its_bare_gateway_alias(
    two_model_settings, tmp_path
):
    (tmp_path / "acme-gateway-key").write_text("sk-acme-secret")

    model = resolve_tenant_chat_model(
        _record(model="openai:claude-eu-large"), settings=two_model_settings
    )

    assert model.model_name == "claude-eu-large"


def test_a_model_setting_outside_the_residencys_list_fails_closed_on_read(two_model_settings):
    """Read side (#105): a stored `model` the allow-list does not (or no longer) allow for the
    tenant's residency is refused exactly like an unlisted deployment default -- never silently
    replaced by the default."""
    with pytest.raises(ModelNotAllowedForResidency):
        resolve_tenant_chat_model(_record(model="claude"), settings=two_model_settings)


def test_a_tenant_that_changes_its_model_gets_the_new_model_on_the_same_client(
    two_model_settings, tmp_path
):
    """The per-tenant cache keeps the tenant's connection (its credential is the resource worth
    not duplicating), never its momentary model choice: a changed setting takes effect on the
    next resolution."""
    (tmp_path / "acme-gateway-key").write_text("sk-acme-secret")
    tenant_id = uuid.uuid4()

    before = resolve_tenant_chat_model(
        _record(model="claude-eu", tenant_id=tenant_id), settings=two_model_settings
    )
    after = resolve_tenant_chat_model(
        _record(model="claude-eu-large", tenant_id=tenant_id), settings=two_model_settings
    )

    assert (before.model_name, after.model_name) == ("claude-eu", "claude-eu-large")
    assert before.provider.client is after.provider.client


def test_missing_gateway_credential_alias_fails_closed(settings):
    """The credential resolver's own typed error (#52) surfaces unchanged -- resolution here
    doesn't mask it behind a different exception."""
    with pytest.raises(GatewayCredentialUnavailable):
        resolve_tenant_chat_model(_record(alias=None), settings=settings)


def test_missing_gateway_credential_file_fails_closed(settings):
    with pytest.raises(GatewayCredentialUnavailable):
        resolve_tenant_chat_model(_record(alias="no-such-file"), settings=settings)


@pytest.mark.parametrize("residency", [None, "", "mars"])
def test_unresolved_residency_fails_closed(settings, residency):
    """A tenant with no usable residency never inherits the deployment's (ADR-0008)."""
    with pytest.raises(ResidencyUnresolved):
        resolve_tenant_chat_model(_record(residency=residency), settings=settings)


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
    """Defense-in-depth: `Settings` itself now refuses to *construct* without
    `LITELLM_BASE_URL` (ADR-0009, ai-app-starter#7, `app.config.Settings._require_gateway_
    configured`), so this constructs a valid `Settings` and then clears the field to simulate a
    value that somehow became unset later (e.g. a stale/attacker-controlled object) -- proving
    `build_tenant_chat_model` has its own, independent guard and never trusts `Settings`
    construction as its only line of defense."""
    settings_without_gateway = Settings(
        database_url="postgresql+asyncpg://app:app@localhost:5432/app",
        gateway_credentials_dir=str(tmp_path),
        litellm_base_url="http://litellm:4000",
    )
    settings_without_gateway.litellm_base_url = None
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


def test_resolve_tenant_embedding_client_uses_the_tenants_own_credential(settings, tmp_path):
    (tmp_path / "acme-gateway-key").write_text("sk-acme-secret")

    client = resolve_tenant_embedding_client(_record(), settings=settings)
    assert client.api_key == "sk-acme-secret"


@pytest.mark.parametrize("residency", [None, "mars"])
def test_resolve_tenant_embedding_client_fails_closed_with_no_resolvable_residency(
    settings, residency
):
    """#61: the embedding path fails closed on an unresolvable residency exactly like the chat
    path does -- it must never fall back to a default embedding endpoint."""
    with pytest.raises(ResidencyUnresolved):
        resolve_tenant_embedding_client(_record(residency=residency), settings=settings)


def test_resolve_tenant_embedding_client_fails_closed_with_no_credential_alias(settings):
    with pytest.raises(GatewayCredentialUnavailable):
        resolve_tenant_embedding_client(_record(alias=None), settings=settings)
