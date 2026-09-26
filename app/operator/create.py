"""The `create` command (Spec 9 / #70 pooled, #71 dedicated, ADR-0010): provisions a tenant --
pooled or dedicated -- in one idempotent run, replacing `scripts/seed.py`.

`create_tenant` performs, in order, against the one `AsyncConnection` the operator CLI's `_run`
already opened (owner role, `app.migration_settings`): (1) the control-plane record (isolation
tier defaults to `'pooled'`, per migration 0005; for `isolation_tier="dedicated"` a fresh alias
is minted and recorded), (2) the admin identity (ADR-0003) -- always here, in the pooled/control
database, since identity resolution always reads that copy
(`app/repositories/control.py`) regardless of a tenant's own isolation tier, (3) a minted gateway
credential written to a secret file and recorded by alias (ADR-0009, ADR-0011) -- also unaffected
by isolation tier. Deliberately everything above happens on that single connection/transaction
rather than by calling `app.gateway_provisioning.provision_gateway_credential` (which opens its
own engine and transaction to stay usable standalone, e.g. by a future `rotate` command): sharing
one transaction is what makes a `create` run atomic end to end for the control-plane pieces -- a
failure at any step, including the gateway call, rolls every DB write for this invocation back
together, and only the credential-issuance primitives (`GatewayAdminClient`,
`GatewayCredentialLimits`, `GATEWAY_MODEL_ALIASES_BY_RESIDENCY`,
`generate_gateway_credential_alias`, `write_gateway_credential_file`, `build_admin_client`) are
reused, not that function's own engine-per-call orchestration.

Where the two tiers diverge is the admin membership (#71, ADR-0002): a pooled tenant's first
admin membership is written to this same pooled connection, exactly as before; a dedicated
tenant's can only ever live in its own database (routing sends every one of its requests there,
never to the pooled one -- see `tests/test_tenant_session_routing_integration.py`), so it is
written by `app.operator.dedicated_db.ensure_dedicated_admin_membership` against that database's
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
half-written. `--dedicated-db-admin-url` is only required, and only checked, at the point a fresh
dedicated database actually needs provisioning (see `ensure_dedicated_database`) -- a pooled
`create`, or a re-run against an already-provisioned dedicated tenant, never needs it.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.config import Settings, get_settings
from app.gateway_provisioning import (
    GATEWAY_MODEL_ALIASES_BY_RESIDENCY,
    GatewayAdminClient,
    GatewayCredentialLimits,
    GatewayProvisioningError,
    build_admin_client,
    generate_gateway_credential_alias,
    write_gateway_credential_file,
)
from app.operator.dedicated_db import (
    ensure_dedicated_admin_membership,
    ensure_dedicated_database,
    generate_database_alias,
)
from app.operator.lookup import TenantNotFoundError, resolve_tenant
from app.tenant_settings import TenantSettings

ISOLATION_TIERS = ("pooled", "dedicated")


class UnrecognizedResidencyError(ValueError):
    """`residency` is not a key of `settings.residency_allow_list.residencies` -- rejected before
    any write."""


class UnrecognizedModelError(ValueError):
    """`model` is not in the model allow-list for the requested residency -- rejected before any
    write."""


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


def _validate_residency(residency: str, settings: Settings) -> None:
    allow_list = settings.residency_allow_list
    assert allow_list is not None  # set by Settings construction
    if residency not in allow_list.residencies:
        raise UnrecognizedResidencyError(
            f"unrecognized residency {residency!r}; known residencies: "
            f"{sorted(allow_list.residencies)} (see settings.residency_allow_list)"
        )


def _validate_model(model: str | None, residency: str, settings: Settings) -> None:
    if model is None:
        return
    allow_list = settings.residency_allow_list
    assert allow_list is not None  # set by Settings construction
    allowed = allow_list.model_aliases(residency)
    if model not in allowed:
        raise UnrecognizedModelError(
            f"unrecognized model {model!r} for residency {residency!r}; allowed models: "
            f"{sorted(allowed)} (see settings.residency_allow_list)"
        )


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

    # Validated before any write, in this order, per acceptance criteria: an unrecognized
    # residency, model-allow-list, or isolation-tier selection must reject before the
    # control-plane record, the credential, or the membership is touched.
    _validate_residency(residency, settings)
    _validate_model(model, residency, settings)
    _validate_isolation_tier(isolation_tier)
    tenant_settings = TenantSettings(model=model) if model is not None else None
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
        # Needed even to *read* control.tenants below: FORCE ROW LEVEL SECURITY applies to
        # app_owner too, and this may be a fresh connection/transaction (e.g. a re-run in a new
        # process) that never set this transaction-local setting.
        await conn.execute(
            text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": str(tenant_id)}
        )
        row = (
            (
                await conn.execute(
                    text(
                        "SELECT residency, isolation_tier, database_alias FROM control.tenants "
                        "WHERE tenant_id = :tid"
                    ),
                    {"tid": tenant_id},
                )
            )
            .mappings()
            .one()
        )
        if row["residency"] != residency:
            raise TenantConflictError(
                f"tenant {tenant_name!r} already exists with residency {row['residency']!r}; "
                f"cannot reconcile it with the requested residency {residency!r}"
            )
        if row["isolation_tier"] != isolation_tier:
            raise TenantConflictError(
                f"tenant {tenant_name!r} already exists with isolation tier "
                f"{row['isolation_tier']!r}; cannot reconcile it with the requested isolation "
                f"tier {isolation_tier!r}"
            )
        control_plane_outcome = "already exists"
        database_alias = row["database_alias"]
    else:
        tenant_id = uuid.uuid4()
        database_alias = (
            generate_database_alias(tenant_id) if isolation_tier == "dedicated" else None
        )
        # Satisfies FORCE ROW LEVEL SECURITY on `tenants`/`control.tenants` even for the owner
        # role, exactly as the retired seed script had to.
        await conn.execute(
            text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": str(tenant_id)}
        )
        await conn.execute(
            text(
                "INSERT INTO tenants (id, name, settings) "
                "VALUES (:id, :name, CAST(:settings AS jsonb))"
            ),
            {"id": tenant_id, "name": tenant_name, "settings": tenant_settings_json},
        )
        await conn.execute(
            text(
                "INSERT INTO control.tenants (tenant_id, residency, isolation_tier, "
                "database_alias) VALUES (:tid, :residency, :tier, :alias)"
            ),
            {
                "tid": tenant_id,
                "residency": residency,
                "tier": isolation_tier,
                "alias": database_alias,
            },
        )
        control_plane_outcome = "created"

    # `app.tenant_id` is already set on `conn` on both paths above -- the existing-tenant branch
    # sets it to read `control.tenants`, the fresh-tenant branch to write it.

    # Admin identity + first membership (ADR-0003). `control.identities` has no tenant_id/RLS
    # (0003) -- the upsert needs no tenant context, only the membership insert below does.
    identity_id = (
        await conn.execute(
            text(
                "INSERT INTO control.identities (id, issuer, subject, display_name, email) "
                "VALUES (:id, :issuer, :subject, :email, :email) "
                "ON CONFLICT (issuer, subject) DO UPDATE SET issuer = EXCLUDED.issuer "
                "RETURNING id"
            ),
            {"id": uuid.uuid4(), "issuer": issuer, "subject": subject, "email": admin_email},
        )
    ).scalar_one()

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
        membership_outcome = await ensure_dedicated_admin_membership(
            owner_dsn=dedicated.owner_dsn,
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            tenant_settings_json=tenant_settings_json,
            identity_id=identity_id,
            issuer=issuer,
            subject=subject,
            admin_email=admin_email,
        )
    else:
        existing_membership = (
            await conn.execute(
                text("SELECT id FROM memberships WHERE tenant_id = :tid AND identity_id = :iid"),
                {"tid": tenant_id, "iid": identity_id},
            )
        ).first()
        if existing_membership is None:
            await conn.execute(
                text(
                    "INSERT INTO memberships (tenant_id, identity_id, role) "
                    "VALUES (:tid, :iid, 'admin')"
                ),
                {"tid": tenant_id, "iid": identity_id},
            )
            membership_outcome = "created"
        else:
            membership_outcome = "already exists"

    # Gateway credential (Spec 7 / #53, ADR-0009, ADR-0011).
    alias_row = (
        await conn.execute(
            text("SELECT gateway_credential_alias FROM control.tenants WHERE tenant_id = :tid"),
            {"tid": tenant_id},
        )
    ).first()
    existing_alias = alias_row[0] if alias_row and alias_row[0] else None

    if existing_alias is not None:
        gateway_outcome = "already provisioned"
        alias = existing_alias
    else:
        models = GATEWAY_MODEL_ALIASES_BY_RESIDENCY.get(residency)
        if models is None:
            raise GatewayProvisioningError(
                f"no gateway model aliases configured for residency {residency!r} "
                f"(known: {sorted(GATEWAY_MODEL_ALIASES_BY_RESIDENCY)})"
            )
        owns_admin_client = admin_client is None
        admin_client = admin_client or build_admin_client(settings)
        try:
            credential = await admin_client.mint_key(
                tenant_id=tenant_id,
                limits=GatewayCredentialLimits(
                    spend_ceiling_usd=settings.gateway_default_spend_ceiling_usd,
                    budget_reset_period=settings.gateway_default_budget_reset_period,
                    requests_per_minute=settings.gateway_default_requests_per_minute,
                    tokens_per_minute=settings.gateway_default_tokens_per_minute,
                ),
                models=models,
            )
            alias = generate_gateway_credential_alias(tenant_id)
            write_gateway_credential_file(alias, credential, settings=settings)
            await conn.execute(
                text(
                    "UPDATE control.tenants SET gateway_credential_alias = :alias "
                    "WHERE tenant_id = :tid"
                ),
                {"alias": alias, "tid": tenant_id},
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
