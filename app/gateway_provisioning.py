"""Mint and revoke a tenant's gateway credential (Spec 7 / #53, ADR-0009, ADR-0011, #114).

Two plain, importable functions -- `provision_gateway_credential` and
`revoke_gateway_credential` -- with no operator-command wrapper of their own, kept usable
standalone (e.g. a future `rotate` command). Both accept an already-open owner-role
`AsyncConnection` (`conn=`) so a caller with its own transaction -- the operator tool's `create`
command (Spec 9 / #70, `app/operator/create.py`) needs the control-plane write, the membership
write, and the credential mint to share one transaction; `erase` (`app/operator/erase.py`) needs
the same for revocation -- can join it instead of this module opening a second one; a caller with
no transaction of its own (a future `rotate` command, a test) can instead pass `owner_engine=` (an
already-built engine this module opens its own transaction on) or nothing at all (this module
builds and disposes its own engine via `app.db.lifecycle.owner_engine`, from
`app.migration_settings`). Exactly one of the three ever applies per call: see `_owner_connection`
below. The admin-client lifecycle (build if not given, close if owned) lives once here too,
reused by revocation the same way.

Provisioning does three things, in order, for a given tenant id:

1. Calls the gateway's administrative interface (`GatewayAdminClient`, LiteLLM's `/key/generate`)
   to mint a credential carrying that tenant's spend ceiling and reset period, its request/token
   rate limit, and a model list restricted to its residency's aliases.
2. Writes the minted credential to the tenant's secret file (`app.gateway_credentials`'s own
   `gateway_credentials_dir`), named by a freshly generated alias.
3. Records that alias in the control plane (`control.tenants.gateway_credential_alias`, #52),
   through `app.repositories.control.ControlRepository` -- the only module that issues SQL
   against the control schema (#114) and the owner role's one write path (#12) onto it.

Revocation reverses steps 2 and 3 and calls the gateway's own revoke operation once. A tenant
with no credential recorded is a no-op, not an error: erasure and rotation both need to call
this on a tenant that may already have been revoked.

The gateway is never reached over a real network in tests: `GatewayAdminClient` is a thin HTTP
wrapper a test replaces with one built on `httpx.MockTransport`, or with a hand-written fake
passed as `admin_client`.
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import httpx
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from app.config import Settings, get_settings
from app.db.lifecycle import owner_engine as _build_owner_engine_cm
from app.gateway_credentials import GatewayCredentialUnavailable, read_gateway_credential
from app.migration_settings import get_migration_settings
from app.repositories.control import ControlRepository

# There used to be a second, hand-maintained copy of the per-residency gateway model aliases
# here (`GATEWAY_MODEL_ALIASES_BY_RESIDENCY`, a Python literal mirroring `config/residency.toml`'s
# own `models` entries) -- a freshly minted key's usable models could drift from the allow-list a
# tenant's own model choice is validated against (`app.llm.validate_model_for_residency`). Removed
# (#85, spec A4 / #111): `provision_gateway_credential` below now takes the models straight from
# `settings.residency_allow_list.model_aliases(residency)` (`app.residency.ResidencyAllowList`),
# the exact same object and method `app.llm`/`app.operator.create` read -- one edit to
# `config/residency.toml` changes what is both allowed and mintable, never two.


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


def build_admin_client(settings: Settings) -> GatewayAdminClient:
    """Public (Spec 9 / #70): the operator tool's `create` command builds its own
    `GatewayAdminClient` this same way (a monkeypatchable module-level name a test replaces) when
    the CLI gives it none, then passes that already-built client into
    `provision_gateway_credential` (`admin_client=`) so exactly one client -- built once, closed
    once, by `create_tenant` -- backs the whole call, never a second one this module would build
    for itself."""
    base_url = _require_setting(settings.litellm_base_url, "LITELLM_BASE_URL")
    master_key = settings.litellm_master_key
    if master_key is None:
        raise GatewayProvisioningError(
            "LITELLM_MASTER_KEY must be set to provision or revoke a gateway credential"
        )
    return GatewayAdminClient(base_url=base_url, master_key=master_key.get_secret_value())


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


@asynccontextmanager
async def _owner_connection(
    conn: AsyncConnection | None, owner_engine: AsyncEngine | None
) -> AsyncIterator[AsyncConnection]:
    """Yields an owner-role `AsyncConnection` inside a transaction, from exactly one of three
    sources, in this priority: `conn` (the caller's own already-open transaction -- `create`/
    `erase` share theirs this way, #114) if given; else a transaction opened on `owner_engine` (the
    caller built and disposes the engine itself, only the transaction is this module's); else a
    fresh engine built from `app.migration_settings` via `app.db.lifecycle.owner_engine`, owned
    (and disposed) for the duration of this one call -- the no-caller-transaction-at-all case (a
    future `rotate` command, a standalone script)."""
    if conn is not None:
        yield conn
        return
    if owner_engine is not None:
        async with owner_engine.begin() as owned_conn:
            yield owned_conn
        return
    dsn = get_migration_settings().database_url_migrations.get_secret_value()
    async with _build_owner_engine_cm(dsn) as engine, engine.begin() as owned_conn:
        yield owned_conn


async def provision_gateway_credential(
    tenant_id: UUID,
    *,
    residency: str,
    limits: GatewayCredentialLimits,
    settings: Settings | None = None,
    admin_client: GatewayAdminClient | None = None,
    conn: AsyncConnection | None = None,
    owner_engine: AsyncEngine | None = None,
) -> str:
    """Mint `tenant_id` a gateway credential scoped to `limits` and `residency`'s model aliases,
    write it to a fresh secret file, and record that file's alias in the control plane through
    `ControlRepository.write_gateway_credential_alias`. Returns the alias.

    `admin_client` is the test seam: pass a fake or `httpx.MockTransport`-backed
    `GatewayAdminClient`; left unset, this builds a real one from `Settings` and closes it here
    too. `conn`/`owner_engine` pick the control-plane write's own connection -- see
    `_owner_connection` above; leaving both unset builds and disposes a fresh owner-role engine.
    """
    settings = settings or get_settings()
    allow_list = settings.residency_allow_list
    assert allow_list is not None  # set by Settings construction
    models = allow_list.model_aliases(residency)
    if not models:
        raise GatewayProvisioningError(
            f"no gateway model aliases configured for residency {residency!r} "
            f"(see settings.residency_allow_list)"
        )

    owns_admin_client = admin_client is None
    admin_client = admin_client or build_admin_client(settings)
    try:
        credential = await admin_client.mint_key(tenant_id=tenant_id, limits=limits, models=models)
        alias = generate_gateway_credential_alias(tenant_id)
        write_gateway_credential_file(alias, credential, settings=settings)
        async with _owner_connection(conn, owner_engine) as owner_conn:
            await ControlRepository().write_gateway_credential_alias(owner_conn, tenant_id, alias)
        return alias
    finally:
        if owns_admin_client:
            await admin_client.aclose()


async def revoke_gateway_credential(
    tenant_id: UUID,
    *,
    settings: Settings | None = None,
    admin_client: GatewayAdminClient | None = None,
    conn: AsyncConnection | None = None,
    owner_engine: AsyncEngine | None = None,
) -> bool:
    """Revoke `tenant_id`'s gateway credential: calls the gateway's revoke operation once,
    removes the per-alias secret file, and clears the alias from the control plane through
    `ControlRepository`.

    Returns `True` if a credential was actually revoked, `False` if none was recorded -- a
    tenant with no alias, or whose secret file is already gone, is a no-op, not an error. A
    second call for the same tenant always returns `False` and never calls the gateway again.
    `conn`/`owner_engine` are the same test/composition seams `provision_gateway_credential` takes
    -- see `_owner_connection`.
    """
    settings = settings or get_settings()
    owns_admin_client = admin_client is None
    try:
        async with _owner_connection(conn, owner_engine) as owner_conn:
            alias = await ControlRepository().read_gateway_credential_alias(owner_conn, tenant_id)
        if alias is None:
            return False

        try:
            credential = read_gateway_credential(alias, settings=settings)
        except GatewayCredentialUnavailable:
            credential = None  # the file is already gone; still revoke at the gateway if we can

        if credential is not None:
            admin_client = admin_client or build_admin_client(settings)
            await admin_client.revoke_key(credential.get_secret_value())

        remove_gateway_credential_file(alias, settings=settings)
        async with _owner_connection(conn, owner_engine) as owner_conn:
            await ControlRepository().write_gateway_credential_alias(owner_conn, tenant_id, None)
        return True
    finally:
        if owns_admin_client and admin_client is not None:
            await admin_client.aclose()
