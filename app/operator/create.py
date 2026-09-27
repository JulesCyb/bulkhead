"""The `create` command (Spec 9 / #70 pooled, #71 dedicated, ADR-0010, #114): provisions a tenant
-- pooled or dedicated -- in one idempotent run, replacing `scripts/seed.py`.

`create_tenant` is a composition over `app.repositories.control.ControlRepository` and
`app.gateway_provisioning.provision_gateway_credential`, against the one `AsyncConnection` the
operator CLI's `_run` already opened (owner role, `app.migration_settings`): (1) the
control-plane record (isolation tier defaults to `'pooled'`, per migration 0005; for
`isolation_tier="dedicated"` a fresh alias is minted and recorded, via
`ControlRepository.create_tenant_record`), (2) the admin identity (ADR-0003, via
`IdentityRepository.upsert`) -- always here, in the pooled/control database, since identity
resolution always reads that copy (`app/repositories/control.py`) regardless of a tenant's own
isolation tier, (3) a minted gateway credential written to a secret file and recorded by alias
(ADR-0009, ADR-0011), via `provision_gateway_credential(..., conn=conn)` -- also unaffected by
isolation tier. Passing this function's own `conn` into `provision_gateway_credential` (rather
than leaving it to open its own engine/transaction, its default when called with neither `conn`
nor `owner_engine`) is what keeps `create` atomic end to end for the control-plane pieces: a
failure at any step, including the gateway call, rolls every DB write for this invocation back
together. The minted credential's usable models come from
`settings.residency_allow_list.model_aliases(residency)` (`app.residency.ResidencyAllowList`,
spec A4 / #85 / #111) -- the same object `_validate_model` below checks the tenant's own model
choice against, so a credential is never minted for a model the allow-list itself would reject;
`provision_gateway_credential` itself raises if that list is empty, so this function does not
re-check it.

Where the two tiers diverge is the admin membership (#71, ADR-0002), written in both cases
through `app.repositories.memberships.ensure_membership` (code review 2026-09-26): a pooled
tenant's first admin membership is written to this same pooled connection; a dedicated
tenant's can only ever live in its own database (routing sends every one of its requests there,
never to the pooled one -- see `tests/test_tenant_session_routing_integration.py`), so it is
written by `app.operator.dedicated_db.ensure_dedicated_membership` against that database's
own owner-role connection instead, after `app.operator.dedicated_db.ensure_dedicated_database`
has provisioned the database itself (CREATE DATABASE, roles and grants, migrated to head) and
written both of its tenant-secret files. Both of those steps live outside this function's own
transaction (a `CREATE DATABASE` cannot run inside one at all) but are themselves idempotent by
construction -- see that module's docstring -- so a re-run against an already-provisioned
dedicated tenant reaches them again harmlessly and does no repeated work.

Idempotency: a tenant is looked up by exact name first (`app.operator.lookup.resolve_tenant`).
Found -- re-running `create` against it performs none of the steps again, reporting each as
already in place; an existing tenant with a different residency, or a different isolation tier
than requested, is a hard error rather than a silent overwrite/retier. Not found -- every step
runs for a fresh tenant id. The identity upsert (`ON CONFLICT (issuer, subject)`) is idempotent on
its own, matching the retired `scripts/seed.py add-membership` pattern; a fresh admin membership
is only inserted if one does not already exist for that (tenant, identity) pair, in whichever
database it belongs to for this tenant's tier.

Residency and an optional per-tenant model override are validated against
`settings.residency_allow_list` (`app.residency.ResidencyAllowList`) before any write happens at
all -- an unrecognized selection never leaves the control plane, a secret file, or a membership
half-written. An optional per-tenant `retention_days` override is validated the same way, against
`settings.max_retention_days` (#84, ADR-0006, `TenantSettings.require_retention_within_cap`).
`--dedicated-db-admin-url` is only required, and only checked, at the point a fresh dedicated
database actually needs provisioning (see `ensure_dedicated_database`) -- a pooled `create`, or a
re-run against an already-provisioned dedicated tenant, never needs it.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncConnection

from app.config import Settings, get_settings
from app.gateway_provisioning import (
    GatewayAdminClient,
    GatewayCredentialLimits,
    build_admin_client,
    provision_gateway_credential,
)
from app.operator.dedicated_db import (
    ensure_dedicated_database,
    ensure_dedicated_membership,
    generate_database_alias,
)
from app.operator.lookup import TenantNotFoundError, resolve_tenant
from app.repositories.control import ControlRepository, IdentityRepository
from app.repositories.memberships import ensure_membership
from app.residency import ResidencyUnresolved
from app.tenant_settings import TenantSettings

ISOLATION_TIERS = ("pooled", "dedicated")


class UnrecognizedResidencyError(ValueError, ResidencyUnresolved):
    """`residency` is not a key of `settings.residency_allow_list.residencies` -- rejected before
    any write. A `ResidencyUnresolved` like every other residency rejection (spec A4's "one
    exception type" rule, code review 2026-09-26), and a `ValueError` for the operator CLI's
    user-facing error handling."""


class UnrecognizedModelError(ValueError, ResidencyUnresolved):
    """`model` is not in the model allow-list for the requested residency -- rejected before any
    write. A `ResidencyUnresolved` and a `ValueError`, like `UnrecognizedResidencyError`."""


class UnrecognizedIsolationTierError(ValueError):
    """`isolation_tier` is not one of `ISOLATION_TIERS` -- rejected before any write."""


class TenantConflictError(ValueError):
    """An existing tenant of this name cannot be reconciled with the requested `create` call --
    a different residency, or a different isolation tier than the one requested."""


@dataclass(frozen=True, slots=True)
class CreateTenantResult:
    tenant_id: UUID
    identity_id: UUID
    control_plane: str  # "created" | "already exists"
    admin_membership: str  # "created" | "already exists"
    gateway_credential: str  # "provisioned" | "already provisioned"
    gateway_credential_alias: str
    isolation_tier: str  # "pooled" | "dedicated"
    database_alias: str | None  # only set for isolation_tier == "dedicated"
    dedicated_database: str | None = None  # "provisioned" | "already provisioned" | None (pooled)

    def render(self) -> str:
        """Byte-identical to what `app.operator.cli`'s retired `_run_create` printed (spec A5 /
        #115): the MCP env vars, the credential alias (and, for a dedicated tenant, its database
        alias), the one-line summary, a blank line, and the worked curl command."""
        lines = [
            f"MCP_TENANT_ID={self.tenant_id}",
            f"MCP_IDENTITY_ID={self.identity_id}",
            f"Gateway credential alias: {self.gateway_credential_alias}",
        ]
        if self.isolation_tier == "dedicated":
            lines.append(f"Database alias: {self.database_alias}")
        lines.append(
            f"control-plane record: {self.control_plane}; "
            f"gateway credential: {self.gateway_credential}; "
            f"admin membership: {self.admin_membership}"
            + (
                f"; dedicated database: {self.dedicated_database}"
                if self.isolation_tier == "dedicated"
                else ""
            )
        )
        lines.append("")
        lines.append(
            f"curl -H 'X-Identity-Id: {self.identity_id}' "
            f"http://localhost:8000/v1/t/{self.tenant_id}/agents/assistant/run ..."
        )
        return "\n".join(lines)

    @property
    def audit_outcome(self) -> str:
        """The one line `app.operator.cli` writes to the operator-action log for this command --
        never printed to stdout (see `render()`), only the audit trail's own summary."""
        return (
            f"ok: tenant {self.tenant_id} "
            f"(control-plane: {self.control_plane}, "
            f"isolation tier: {self.isolation_tier}, "
            f"gateway: {self.gateway_credential}, "
            f"membership: {self.admin_membership})"
        )


def _validate_residency(residency: str, settings: Settings) -> None:
    """Rejects an unrecognized residency before any write, via `route_for` (spec A4 / #111) --
    the operator tool's own `UnrecognizedResidencyError` (a `ValueError`, what the CLI's
    user-facing error handling expects, and itself a `ResidencyUnresolved`) chains the allow-list's
    own `ResidencyUnresolved` rather than re-implementing the lookup against
    `allow_list.residencies` itself.
    """
    allow_list = settings.residency_allow_list
    assert allow_list is not None  # set by Settings construction
    try:
        allow_list.route_for(residency)
    except ResidencyUnresolved as exc:
        raise UnrecognizedResidencyError(
            f"unrecognized residency {residency!r}; known residencies: "
            f"{sorted(allow_list.residencies)} (see settings.residency_allow_list)"
        ) from exc


def _validate_model(model: str | None, residency: str, settings: Settings) -> None:
    """Rejects a model outside `residency`'s allow-list before any write, via `alias_for` (spec
    A4 / #111) -- same reasoning as `_validate_residency` above."""
    if model is None:
        return
    allow_list = settings.residency_allow_list
    assert allow_list is not None  # set by Settings construction
    try:
        allow_list.alias_for(residency, model)
    except ResidencyUnresolved as exc:
        raise UnrecognizedModelError(
            f"unrecognized model {model!r} for residency {residency!r}; allowed models: "
            f"{sorted(allow_list.model_aliases(residency))} (see settings.residency_allow_list)"
        ) from exc


def _validate_isolation_tier(isolation_tier: str) -> None:
    if isolation_tier not in ISOLATION_TIERS:
        raise UnrecognizedIsolationTierError(
            f"unrecognized isolation tier {isolation_tier!r}; known tiers: {ISOLATION_TIERS}"
        )


async def create_tenant(
    conn: AsyncConnection,
    *,
    tenant_name: str,
    residency: str,
    admin_email: str,
    model: str | None = None,
    retention_days: int | None = None,
    isolation_tier: str = "pooled",
    dedicated_db_admin_url: str | None = None,
    issuer: str = "dev-seed",
    subject: str | None = None,
    settings: Settings | None = None,
    admin_client: GatewayAdminClient | None = None,
) -> CreateTenantResult:
    """Provision (or reconcile) a tenant named `tenant_name`. See module docstring for the full
    contract. `admin_client` is the test seam (an `httpx.MockTransport`-backed
    `GatewayAdminClient` or a hand-written fake); left unset, this builds a real one from
    `Settings`, exactly as the retired seed script did. `dedicated_db_admin_url` is only
    consulted when `isolation_tier="dedicated"` and this is a fresh tenant whose database has not
    already been provisioned (see `app.operator.dedicated_db.ensure_dedicated_database`).
    """
    settings = settings or get_settings()
    repo = ControlRepository()

    # Validated before any write, in this order, per acceptance criteria: an unrecognized
    # residency, model-allow-list, isolation-tier, or retention-days-over-the-cap (#84) selection
    # must reject before the control-plane record, the credential, or the membership is touched.
    _validate_residency(residency, settings)
    _validate_model(model, residency, settings)
    _validate_isolation_tier(isolation_tier)
    tenant_settings = (
        TenantSettings(model=model, retention_days=retention_days)
        if model is not None or retention_days is not None
        else None
    )
    if tenant_settings is not None:
        # The write-side half of the retention cap (#84) lives on the model itself; the cap is
        # passed in because `TenantSettings` cannot see process configuration.
        tenant_settings.require_retention_within_cap(max_days=settings.max_retention_days)
    tenant_settings_json = json.dumps(
        tenant_settings.model_dump(exclude_none=True) if tenant_settings else {}
    )

    subject = subject or admin_email

    try:
        existing = await resolve_tenant(conn, tenant_name)
    except TenantNotFoundError:
        existing = None
    # AmbiguousTenantNameError is intentionally left to propagate: create cannot safely pick one
    # of several same-named tenants to reconcile against.

    if existing is not None:
        tenant_id = existing.tenant_id
        record = await repo.get_record(conn, tenant_id)
        if record.residency != residency:
            raise TenantConflictError(
                f"tenant {tenant_name!r} already exists with residency {record.residency!r}; "
                f"cannot reconcile it with the requested residency {residency!r}"
            )
        if record.isolation_tier != isolation_tier:
            raise TenantConflictError(
                f"tenant {tenant_name!r} already exists with isolation tier "
                f"{record.isolation_tier!r}; cannot reconcile it with the requested isolation "
                f"tier {isolation_tier!r}"
            )
        control_plane_outcome = "already exists"
        database_alias = record.database_alias
        existing_alias = record.gateway_credential_alias
    else:
        tenant_id = uuid.uuid4()
        database_alias = (
            generate_database_alias(tenant_id) if isolation_tier == "dedicated" else None
        )
        await repo.create_tenant_record(
            conn,
            tenant_id,
            name=tenant_name,
            residency=residency,
            isolation_tier=isolation_tier,
            database_alias=database_alias,
            settings_json=tenant_settings_json,
        )
        control_plane_outcome = "created"
        existing_alias = None

    # Admin identity + first membership (ADR-0003). `control.identities` has no tenant_id/RLS
    # (0003), so the upsert needs no tenant context. The membership does (forced RLS): on the
    # pooled path it is the one `get_record`/`create_tenant_record` above already set on this
    # transaction through the control repository's forced-RLS helper; on the dedicated path
    # `ensure_dedicated_membership` sets it against the tenant's own database.
    identity_id = await IdentityRepository().upsert(
        conn, id=uuid.uuid4(), issuer=issuer, subject=subject, email=admin_email
    )

    dedicated_database_outcome: str | None = None
    if isolation_tier == "dedicated":
        # A dedicated tenant's membership can only ever live in its own database (routing never
        # sends its requests to the pooled one) -- provision that database first, then write the
        # membership there instead of on `conn`.
        assert database_alias is not None  # enforced by migration 0005's CHECK constraint
        dedicated = await ensure_dedicated_database(
            alias=database_alias, admin_url=dedicated_db_admin_url
        )
        dedicated_database_outcome = dedicated.outcome
        membership_outcome = await ensure_dedicated_membership(
            owner_dsn=dedicated.owner_dsn,
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            tenant_settings_json=tenant_settings_json,
            identity_id=identity_id,
            issuer=issuer,
            subject=subject,
            email=admin_email,
            role="admin",
        )
    else:
        membership_outcome = await ensure_membership(
            conn, tenant_id=tenant_id, identity_id=identity_id, role="admin"
        )

    # Gateway credential (Spec 7 / #53, ADR-0009, ADR-0011): minted, written to a secret file, and
    # recorded by alias in one call -- `provision_gateway_credential` itself raises if the
    # residency's model allow-list is empty, so this never re-checks that.
    if existing_alias is not None:
        gateway_outcome = "already provisioned"
        alias = existing_alias
    else:
        owns_admin_client = admin_client is None
        admin_client = admin_client or build_admin_client(settings)
        try:
            alias = await provision_gateway_credential(
                tenant_id,
                residency=residency,
                limits=GatewayCredentialLimits(
                    spend_ceiling_usd=settings.gateway_default_spend_ceiling_usd,
                    budget_reset_period=settings.gateway_default_budget_reset_period,
                    requests_per_minute=settings.gateway_default_requests_per_minute,
                    tokens_per_minute=settings.gateway_default_tokens_per_minute,
                ),
                settings=settings,
                admin_client=admin_client,
                conn=conn,
            )
            gateway_outcome = "provisioned"
        finally:
            if owns_admin_client:
                await admin_client.aclose()

    return CreateTenantResult(
        tenant_id=tenant_id,
        identity_id=identity_id,
        control_plane=control_plane_outcome,
        admin_membership=membership_outcome,
        gateway_credential=gateway_outcome,
        gateway_credential_alias=alias,
        isolation_tier=isolation_tier,
        database_alias=database_alias,
        dedicated_database=dedicated_database_outcome,
    )
