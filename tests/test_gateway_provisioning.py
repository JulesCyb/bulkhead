"""Mint/revoke a tenant's gateway credential (Spec 7 / #53): the gateway-call shaping, the
error mapping, and the secret-file half, exercised without a real database or a real network
call -- `GatewayAdminClient` is built on `httpx.MockTransport`, so this is the "ASGI application
plus Settings, gateway call faked" seam. The control-plane read/write half is covered by the
embedded-Postgres tests in tests/test_gateway_provisioning_integration.py.

Since #85 / spec A4 / #111: the minted credential's usable models come from
`settings.residency_allow_list.model_aliases(residency)` (`app.residency.ResidencyAllowList`),
never a second, hand-maintained Python literal -- the retired `GATEWAY_MODEL_ALIASES_BY_RESIDENCY`
must never come back.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from app.config import Settings
from app.gateway_credentials import GatewayCredentialUnavailable
from app.gateway_provisioning import (
    GatewayAdminClient,
    GatewayCredentialLimits,
    GatewayProvisioningError,
    generate_gateway_credential_alias,
    provision_gateway_credential,
    remove_gateway_credential_file,
    revoke_gateway_credential,
    write_gateway_credential_file,
)
from app.llm import validate_model_for_residency
from app.residency import ResidencyAllowList


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_url="postgresql+asyncpg://app:app@localhost:5432/app",
        gateway_credentials_dir=str(tmp_path),
        litellm_base_url="http://litellm.internal:4000",
        litellm_master_key="sk-master-test",
    )


LIMITS = GatewayCredentialLimits(
    spend_ceiling_usd=25.0,
    budget_reset_period="30d",
    requests_per_minute=60,
    tokens_per_minute=100_000,
)


def _admin_client(handler) -> GatewayAdminClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(
        base_url="http://litellm.internal:4000",
        transport=transport,
        headers={"Authorization": "Bearer sk-master-test"},
    )
    return GatewayAdminClient(
        base_url="http://litellm.internal:4000",
        master_key="sk-master-test",
        http_client=http_client,
    )


# --- GatewayAdminClient.mint_key -----------------------------------------------------------


async def test_mint_key_posts_budget_rate_limit_and_models_to_key_generate():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["auth"] = request.headers["authorization"]
        captured["body"] = httpx.Request("POST", request.url, content=request.content).content
        import json

        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"key": "sk-minted"})

    tenant_id = uuid.uuid4()
    client = _admin_client(handler)
    key = await client.mint_key(
        tenant_id=tenant_id, limits=LIMITS, models=("claude-eu", "embeddings")
    )

    assert key == "sk-minted"
    assert captured["path"] == "/key/generate"
    assert captured["auth"] == "Bearer sk-master-test"
    body = captured["json"]
    assert body["max_budget"] == 25.0
    assert body["budget_duration"] == "30d"
    assert body["rpm_limit"] == 60
    assert body["tpm_limit"] == 100_000
    assert body["models"] == ["claude-eu", "embeddings"]
    assert body["metadata"]["tenant_id"] == str(tenant_id)


async def test_mint_key_raises_typed_error_on_non_2xx_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad budget")

    client = _admin_client(handler)
    with pytest.raises(GatewayProvisioningError):
        await client.mint_key(tenant_id=uuid.uuid4(), limits=LIMITS, models=("claude-eu",))


async def test_mint_key_raises_typed_error_when_response_has_no_key():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    client = _admin_client(handler)
    with pytest.raises(GatewayProvisioningError):
        await client.mint_key(tenant_id=uuid.uuid4(), limits=LIMITS, models=("claude-eu",))


# --- GatewayAdminClient.revoke_key ----------------------------------------------------------


async def test_revoke_key_posts_the_key_to_key_delete():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        import json

        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"deleted_keys": ["sk-minted"]})

    client = _admin_client(handler)
    await client.revoke_key("sk-minted")
    assert captured["path"] == "/key/delete"
    assert captured["json"] == {"keys": ["sk-minted"]}


async def test_revoke_key_raises_typed_error_on_non_2xx_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="no such key")

    client = _admin_client(handler)
    with pytest.raises(GatewayProvisioningError):
        await client.revoke_key("sk-gone")


# --- Model list per residency ---------------------------------------------------------------


async def test_unknown_residency_is_rejected_before_any_gateway_call():
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"key": "sk-minted"})

    with pytest.raises(GatewayProvisioningError):
        await provision_gateway_credential(
            uuid.uuid4(),
            residency="mars",
            limits=LIMITS,
            admin_client=_admin_client(handler),
        )
    assert called is False


def test_eu_and_us_residencies_have_distinct_model_lists(settings):
    allow_list = settings.residency_allow_list
    assert allow_list.model_aliases("eu") != allow_list.model_aliases("us")


# --- Secret-file helpers ----------------------------------------------------------------------


def test_write_then_remove_gateway_credential_file(tmp_path, settings):
    alias = "acme-gw"
    path = write_gateway_credential_file(alias, "sk-acme", settings=settings)
    assert path.read_text() == "sk-acme"
    remove_gateway_credential_file(alias, settings=settings)
    assert not path.exists()


def test_removing_an_already_absent_credential_file_is_not_an_error(settings):
    remove_gateway_credential_file("never-written", settings=settings)  # must not raise


def test_generated_aliases_are_unique_even_for_the_same_tenant():
    tenant_id = uuid.uuid4()
    first = generate_gateway_credential_alias(tenant_id)
    second = generate_gateway_credential_alias(tenant_id)
    assert first != second
    assert str(tenant_id) in first


# --- provision_gateway_credential / revoke_gateway_credential orchestration, DB faked --------


class _FakeControlPlane:
    """Stands in for `ControlRepository`'s own gateway-alias read/write so this module's
    orchestration is testable with no real database -- the real control-plane behavior
    (owner-only write, RLS, the resolver reading it back) is covered by the embedded-Postgres
    integration test.
    """

    def __init__(self) -> None:
        self.aliases: dict[uuid.UUID, str | None] = {}

    async def record(self, tenant_id: uuid.UUID, alias: str | None) -> None:
        self.aliases[tenant_id] = alias

    async def read(self, tenant_id: uuid.UUID) -> str | None:
        return self.aliases.get(tenant_id)


@pytest.fixture
def control_plane(monkeypatch):
    fake = _FakeControlPlane()
    from app.repositories.control import ControlRepository

    async def fake_write(self, conn, tenant_id, alias):
        await fake.record(tenant_id, alias)

    async def fake_read(self, conn, tenant_id):
        return await fake.read(tenant_id)

    monkeypatch.setattr(ControlRepository, "write_gateway_credential_alias", fake_write)
    monkeypatch.setattr(ControlRepository, "read_gateway_credential_alias", fake_read)
    return fake


async def test_provision_writes_file_and_records_alias(settings, control_plane, tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"key": "sk-acme"})

    tenant_id = uuid.uuid4()
    alias = await provision_gateway_credential(
        tenant_id,
        residency="eu",
        limits=LIMITS,
        settings=settings,
        admin_client=_admin_client(handler),
        conn=object(),
    )
    assert (tmp_path / alias).read_text() == "sk-acme"
    assert control_plane.aliases[tenant_id] == alias


async def test_model_added_to_the_allow_list_is_both_allowed_and_mintable_with_no_second_edit(
    control_plane, tmp_path
):
    """#85 (spec A4 / #111): a model added to a residency's allow-list in the TOML/in-memory data
    is both allowed by model resolution (`app.llm.validate_model_for_residency`) and included in
    a freshly minted credential's usable models -- one edit to the allow-list data, never a
    second, hand-maintained Python literal (the retired `GATEWAY_MODEL_ALIASES_BY_RESIDENCY`) to
    keep in sync with it.
    """
    allow_list = ResidencyAllowList.from_data(
        {
            "residency": {
                "eu": {
                    "model_host_patterns": ["litellm.internal"],
                    "embedding_endpoint": "https://litellm.internal/v1",
                    "trace_sink_host": "eu.cloud.langfuse.com",
                    "models": ["claude-eu", "embeddings", "claude-eu-mini"],
                }
            }
        }
    )
    settings = Settings(
        database_url="postgresql+asyncpg://app:app@localhost:5432/app",
        gateway_credentials_dir=str(tmp_path),
        litellm_base_url="http://litellm.internal:4000",
        litellm_master_key="sk-master-test",
        residency_allow_list=allow_list,
    )

    # Allowed: the freshly added model resolves for its residency with no second edit anywhere.
    assert validate_model_for_residency("claude-eu-mini", "eu", settings=settings) == (
        "claude-eu-mini"
    )

    captured: dict[str, list[str]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["models"] = json.loads(request.content)["models"]
        return httpx.Response(200, json={"key": "sk-acme"})

    await provision_gateway_credential(
        uuid.uuid4(),
        residency="eu",
        limits=LIMITS,
        settings=settings,
        admin_client=_admin_client(handler),
        conn=object(),
    )

    # Mintable: the same freshly added model was actually sent to the gateway as a usable model.
    assert "claude-eu-mini" in captured["models"]


async def test_two_tenants_get_distinct_aliases_and_credentials(settings, control_plane, tmp_path):
    def make_handler(key: str):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"key": key})

        return handler

    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    alias_a = await provision_gateway_credential(
        tenant_a,
        residency="eu",
        limits=LIMITS,
        settings=settings,
        admin_client=_admin_client(make_handler("sk-a")),
        conn=object(),
    )
    alias_b = await provision_gateway_credential(
        tenant_b,
        residency="eu",
        limits=LIMITS,
        settings=settings,
        admin_client=_admin_client(make_handler("sk-b")),
        conn=object(),
    )
    assert alias_a != alias_b
    assert (tmp_path / alias_a).read_text() != (tmp_path / alias_b).read_text()


async def test_revoke_calls_gateway_once_and_removes_the_file(settings, control_plane, tmp_path):
    tenant_id = uuid.uuid4()
    revoke_calls: list[str] = []

    def mint_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"key": "sk-acme"})

    alias = await provision_gateway_credential(
        tenant_id,
        residency="eu",
        limits=LIMITS,
        settings=settings,
        admin_client=_admin_client(mint_handler),
        conn=object(),
    )
    assert (tmp_path / alias).exists()

    def revoke_handler(request: httpx.Request) -> httpx.Response:
        revoke_calls.append(request.url.path)
        return httpx.Response(200, json={"deleted_keys": ["sk-acme"]})

    revoked = await revoke_gateway_credential(
        tenant_id,
        settings=settings,
        admin_client=_admin_client(revoke_handler),
        conn=object(),
    )
    assert revoked is True
    assert revoke_calls == ["/key/delete"]
    assert not (tmp_path / alias).exists()
    assert control_plane.aliases[tenant_id] is None


async def test_second_revoke_is_a_noop_and_never_calls_the_gateway_again(
    settings, control_plane, tmp_path
):
    tenant_id = uuid.uuid4()

    def mint_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"key": "sk-acme"})

    await provision_gateway_credential(
        tenant_id,
        residency="eu",
        limits=LIMITS,
        settings=settings,
        admin_client=_admin_client(mint_handler),
        conn=object(),
    )

    revoke_calls: list[str] = []

    def revoke_handler(request: httpx.Request) -> httpx.Response:
        revoke_calls.append(request.url.path)
        return httpx.Response(200, json={"deleted_keys": ["sk-acme"]})

    first = await revoke_gateway_credential(
        tenant_id,
        settings=settings,
        admin_client=_admin_client(revoke_handler),
        conn=object(),
    )
    second = await revoke_gateway_credential(
        tenant_id,
        settings=settings,
        admin_client=_admin_client(revoke_handler),
        conn=object(),
    )
    assert first is True
    assert second is False
    assert revoke_calls == ["/key/delete"]  # only the first revoke ever reached the gateway


async def test_revoking_a_tenant_with_no_credential_is_a_noop(settings, control_plane):
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"deleted_keys": []})

    revoked = await revoke_gateway_credential(
        uuid.uuid4(), settings=settings, admin_client=_admin_client(handler), conn=object()
    )
    assert revoked is False
    assert called is False


async def test_revoke_handles_a_missing_secret_file_gracefully(settings, control_plane):
    """The alias is recorded but its secret file is already gone (e.g. a prior partial
    failure): revoke must not raise `GatewayCredentialUnavailable`, and still clears the alias.
    """
    tenant_id = uuid.uuid4()
    control_plane.aliases[tenant_id] = "orphaned-alias"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"deleted_keys": []})

    revoked = await revoke_gateway_credential(
        tenant_id, settings=settings, admin_client=_admin_client(handler), conn=object()
    )
    assert revoked is True
    assert control_plane.aliases[tenant_id] is None
    with pytest.raises(GatewayCredentialUnavailable):
        from app.gateway_credentials import read_gateway_credential

        read_gateway_credential("orphaned-alias", settings=settings)


# --- Missing configuration ----------------------------------------------------------------


async def test_provisioning_without_a_base_url_raises_before_any_call(tmp_path):
    """Defense-in-depth: `Settings` itself now refuses to *construct* without
    `LITELLM_BASE_URL` (ADR-0009, ai-app-starter#7, `Settings._require_gateway_configured`), so
    this constructs a valid `Settings` and clears the field afterwards to prove
    `provision_gateway_credential` has its own, independent guard too."""
    settings = Settings(
        database_url="postgresql+asyncpg://app:app@localhost:5432/app",
        gateway_credentials_dir=str(tmp_path),
        litellm_base_url="http://litellm:4000",
        litellm_master_key="sk-master-test",
    )
    settings.litellm_base_url = None
    with pytest.raises(GatewayProvisioningError):
        await provision_gateway_credential(
            uuid.uuid4(), residency="eu", limits=LIMITS, settings=settings
        )


async def test_provisioning_without_a_master_key_raises_before_any_call(tmp_path):
    settings = Settings(
        database_url="postgresql+asyncpg://app:app@localhost:5432/app",
        gateway_credentials_dir=str(tmp_path),
        litellm_base_url="http://litellm.internal:4000",
        litellm_master_key=None,
    )
    with pytest.raises(GatewayProvisioningError):
        await provision_gateway_credential(
            uuid.uuid4(), residency="eu", limits=LIMITS, settings=settings
        )
