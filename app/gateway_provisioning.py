"""Mint and revoke a tenant's gateway credential (Spec 7 / #53, ADR-0009, ADR-0011).

Two plain, importable functions -- `provision_gateway_credential` and
`revoke_gateway_credential` -- with no operator-command wrapper of their own: Spec 9's future
`create`/`suspend`/`erase` operator tool imports and calls these unchanged. Until that tool
exists, `scripts/seed.py` calls `provision_gateway_credential` directly so the documented
quickstart keeps producing a tenant that can actually call the assistant.

Provisioning does three things, in order, for a given tenant id:

1. Calls the gateway's administrative interface (`GatewayAdminClient`, LiteLLM's `/key/generate`)
   to mint a credential carrying that tenant's spend ceiling and reset period, its request/token
   rate limit, and a model list restricted to its residency's aliases.
2. Writes the minted credential to the tenant's secret file (`app.gateway_credentials`'s own
   `gateway_credentials_dir`), named by a freshly generated alias.
3. Records that alias in the control plane (`control.tenants.gateway_credential_alias`, #52),
   through the owner role's write path (#12) -- the only role ever permitted to write there.

Revocation reverses steps 2 and 3 and calls the gateway's own revoke operation once. A tenant
with no credential recorded is a no-op, not an error: erasure and rotation both need to call
this on a tenant that may already have been revoked.

The gateway is never reached over a real network in tests: `GatewayAdminClient` is a thin HTTP
wrapper a test replaces with one built on `httpx.MockTransport`, or with a hand-written fake
passed as `admin_client`.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config import Settings, get_settings
from app.gateway_credentials import GatewayCredentialUnavailable, read_gateway_credential
from app.migration_settings import get_migration_settings

# Mirrors the `model_name` entries in docker/litellm/config.yaml. This is the narrower,
# provisioning-time list a freshly minted key's usable models is restricted to -- distinct from
# the process-wide, per-residency allow-list Spec 7 / #54 adds for validating a tenant's *own*
# `tenants.settings["model"]` choice before a client is built. The two lists describe the same
# gateway configuration from two different call sites and must stay consistent with it.
GATEWAY_MODEL_ALIASES_BY_RESIDENCY: dict[str, tuple[str, ...]] = {
    "eu": ("claude-eu", "embeddings"),
    "us": ("claude", "embeddings"),
}


class GatewayProvisioningError(Exception):
    """The gateway's administrative interface could not mint or revoke a credential: a
    misconfigured deployment (no admin URL/master key, an unknown residency), a non-2xx response,
    or a response this module cannot use (no `key` field). Never raised for, and never masks, a
    control-plane or filesystem failure -- those surface as their own exceptions.
    """


@dataclass(frozen=True)
class GatewayCredentialLimits:
    """What a minted gateway credential is scoped to, independent of any one gateway's own
    parameter names: a budget (CONTEXT.md's term -- spend ceiling and reset period) and a
    request/token rate limit, both distinct from a run limit (app/run_limits.py), which is
    enforced in the application, not at the gateway.
    """

    spend_ceiling_usd: float
    budget_reset_period: str  # a LiteLLM duration string, e.g. "30d", "1mo"
    requests_per_minute: int
    tokens_per_minute: int


class GatewayAdminClient:
    """Thin wrapper over the gateway's administrative HTTP interface (LiteLLM's `/key/generate`,
    `/key/delete`). Every gateway call this module makes goes through here -- a test builds one
    on `httpx.MockTransport`, or replaces the whole client with a hand-written fake; nothing here
    ever reaches a real gateway in a test.
    """

    def __init__(
        self, *, base_url: str, master_key: str, http_client: httpx.AsyncClient | None = None
    ) -> None:
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {master_key}"},
            timeout=10.0,
        )

    async def mint_key(
        self, *, tenant_id: UUID, limits: GatewayCredentialLimits, models: tuple[str, ...]
    ) -> str:
        """Mint a virtual key scoped to `limits` and restricted to `models`. Returns the raw
        credential string. Raises `GatewayProvisioningError` on a non-2xx response or a response
        with no usable `key` field.
        """
        response = await self._client.post(
            "/key/generate",
            json={
                "max_budget": limits.spend_ceiling_usd,
                "budget_duration": limits.budget_reset_period,
                "rpm_limit": limits.requests_per_minute,
                "tpm_limit": limits.tokens_per_minute,
                "models": list(models),
                "metadata": {"tenant_id": str(tenant_id)},
            },
        )
        if response.status_code >= 400:
            raise GatewayProvisioningError(
                f"gateway rejected key/generate for tenant {tenant_id}: "
                f"{response.status_code} {response.text}"
            )
        key = response.json().get("key")
        if not key:
            raise GatewayProvisioningError(
                f"gateway key/generate response for tenant {tenant_id} carried no 'key' field"
            )
        return key

    async def revoke_key(self, credential: str) -> None:
        """Revoke a previously minted key. Raises `GatewayProvisioningError` on a non-2xx
        response."""
        response = await self._client.post("/key/delete", json={"keys": [credential]})
        if response.status_code >= 400:
            raise GatewayProvisioningError(
                f"gateway rejected key/delete: {response.status_code} {response.text}"
            )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _require_setting(value: str | None, name: str) -> str:
    if not value:
        raise GatewayProvisioningError(
            f"{name} must be set to provision or revoke a gateway credential"
        )
    return value


def _build_admin_client(settings: Settings) -> GatewayAdminClient:
    base_url = _require_setting(settings.litellm_base_url, "LITELLM_BASE_URL")
    master_key = settings.litellm_master_key
    if master_key is None:
        raise GatewayProvisioningError(
            "LITELLM_MASTER_KEY must be set to provision or revoke a gateway credential"
        )
    return GatewayAdminClient(base_url=base_url, master_key=master_key.get_secret_value())


def _build_owner_engine() -> AsyncEngine:
    dsn = get_migration_settings().database_url_migrations.get_secret_value()
    return create_async_engine(dsn)


def generate_gateway_credential_alias(tenant_id: UUID) -> str:
    """A fresh alias for `tenant_id`'s gateway credential: unique per call, so re-provisioning
    (rotation) never reuses a still-referenced secret file's name.
    """
    return f"gw-{tenant_id}-{secrets.token_hex(4)}"


def write_gateway_credential_file(alias: str, credential: str, *, settings: Settings) -> Path:
    """Write `credential` to `alias`'s secret file under `settings.gateway_credentials_dir`,
    creating the directory if needed. Mirrors `app.gateway_credentials.read_gateway_credential`'s
    own path resolution exactly.
    """
    directory = Path(settings.gateway_credentials_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / alias
    path.write_text(credential)
    try:
        path.chmod(0o600)
    except OSError:
        pass  # best-effort on filesystems that don't support POSIX permissions (e.g. some CI)
    return path


def remove_gateway_credential_file(alias: str, *, settings: Settings) -> None:
    """Remove `alias`'s secret file if present. A missing file is not an error -- revoking an
    already-revoked credential must be a no-op, not a failure.
    """
    path = Path(settings.gateway_credentials_dir) / alias
    path.unlink(missing_ok=True)


async def _record_alias_in_control_plane(
    tenant_id: UUID, alias: str | None, *, owner_engine: AsyncEngine
) -> None:
    """Write `alias` (or clear it, `None`) to `control.tenants.gateway_credential_alias` as the
    owner role -- the only role ever permitted to write there (#12). `FORCE ROW LEVEL SECURITY`
    (0001/0002) applies to the owner role too, so this still sets `app.tenant_id` to the target
    tenant first, exactly as a real provisioning step must. `ON CONFLICT` covers both a tenant
    provisioned for the first time (no `control.tenants` row yet) and re-provisioning/rotation of
    one that already has a row.
    """
    async with owner_engine.begin() as conn:
        await conn.execute(
            text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": str(tenant_id)}
        )
        await conn.execute(
            text(
                "INSERT INTO control.tenants (tenant_id, gateway_credential_alias) "
                "VALUES (:tid, :alias) "
                "ON CONFLICT (tenant_id) DO UPDATE "
                "SET gateway_credential_alias = EXCLUDED.gateway_credential_alias"
            ),
            {"tid": tenant_id, "alias": alias},
        )


async def _read_alias_from_control_plane(
    tenant_id: UUID, *, owner_engine: AsyncEngine
) -> str | None:
    """The alias currently recorded for `tenant_id`, or `None` if no `control.tenants` row
    exists for it yet (never provisioned) or its alias column is unset (provisioned but revoked).
    """
    async with owner_engine.begin() as conn:
        await conn.execute(
            text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": str(tenant_id)}
        )
        row = (
            await conn.execute(
                text("SELECT gateway_credential_alias FROM control.tenants WHERE tenant_id = :tid"),
                {"tid": tenant_id},
            )
        ).first()
    return row[0] if row and row[0] else None


async def provision_gateway_credential(
    tenant_id: UUID,
    *,
    residency: str,
    limits: GatewayCredentialLimits,
    settings: Settings | None = None,
    admin_client: GatewayAdminClient | None = None,
    owner_engine: AsyncEngine | None = None,
) -> str:
    """Mint `tenant_id` a gateway credential scoped to `limits` and `residency`'s model aliases,
    write it to a fresh secret file, and record that file's alias in the control plane. Returns
    the alias.

    `admin_client`/`owner_engine` are the test seams: pass a fake or `httpx.MockTransport`-backed
    `GatewayAdminClient` and/or an engine already pointed at a real owner-role connection. Left
    unset, this builds real ones from `Settings`/`app.migration_settings`, exactly as the seed
    script and a future operator tool do.
    """
    settings = settings or get_settings()
    models = GATEWAY_MODEL_ALIASES_BY_RESIDENCY.get(residency)
    if models is None:
        raise GatewayProvisioningError(
            f"no gateway model aliases configured for residency {residency!r} "
            f"(known: {sorted(GATEWAY_MODEL_ALIASES_BY_RESIDENCY)})"
        )

    owns_admin_client = admin_client is None
    admin_client = admin_client or _build_admin_client(settings)
    owns_owner_engine = owner_engine is None
    owner_engine = owner_engine or _build_owner_engine()
    try:
        credential = await admin_client.mint_key(tenant_id=tenant_id, limits=limits, models=models)
        alias = generate_gateway_credential_alias(tenant_id)
        write_gateway_credential_file(alias, credential, settings=settings)
        await _record_alias_in_control_plane(tenant_id, alias, owner_engine=owner_engine)
        return alias
    finally:
        if owns_admin_client:
            await admin_client.aclose()
        if owns_owner_engine:
            await owner_engine.dispose()


async def revoke_gateway_credential(
    tenant_id: UUID,
    *,
    settings: Settings | None = None,
    admin_client: GatewayAdminClient | None = None,
    owner_engine: AsyncEngine | None = None,
) -> bool:
    """Revoke `tenant_id`'s gateway credential: calls the gateway's revoke operation once,
    removes the per-alias secret file, and clears the alias from the control plane.

    Returns `True` if a credential was actually revoked, `False` if none was recorded -- a
    tenant with no alias, or whose secret file is already gone, is a no-op, not an error. A
    second call for the same tenant always returns `False` and never calls the gateway again.
    """
    settings = settings or get_settings()
    owns_owner_engine = owner_engine is None
    owner_engine = owner_engine or _build_owner_engine()
    owns_admin_client = admin_client is None
    try:
        alias = await _read_alias_from_control_plane(tenant_id, owner_engine=owner_engine)
        if alias is None:
            return False

        try:
            credential = read_gateway_credential(alias, settings=settings)
        except GatewayCredentialUnavailable:
            credential = None  # the file is already gone; still revoke at the gateway if we can

        if credential is not None:
            admin_client = admin_client or _build_admin_client(settings)
            await admin_client.revoke_key(credential.get_secret_value())

        remove_gateway_credential_file(alias, settings=settings)
        await _record_alias_in_control_plane(tenant_id, None, owner_engine=owner_engine)
        return True
    finally:
        if owns_admin_client and admin_client is not None:
            await admin_client.aclose()
        if owns_owner_engine:
            await owner_engine.dispose()
