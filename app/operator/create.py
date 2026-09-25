"""The `create` command (Spec 9 / #70, ADR-0010): provisions a pooled tenant in one idempotent
run, replacing `scripts/seed.py`.

`create_tenant` performs, in order, against the one `AsyncConnection` the operator CLI's `_run`
already opened (owner role, `app.migration_settings`): (1) the control-plane record (isolation
tier defaults to `'pooled'`, per migration 0005; database alias stays `NULL` -- a dedicated
tier is #71's job, out of scope here), (2) the first admin identity and membership (ADR-0003),
(3) a minted gateway credential written to a secret file and recorded by alias (ADR-0009,
ADR-0011). Deliberately everything happens on that single connection/transaction rather than by
calling `app.gateway_provisioning.provision_gateway_credential` (which opens its own engine and
transaction to stay usable standalone, e.g. by a future `rotate` command): sharing one
transaction is what makes a `create` run atomic end to end -- a failure at any step, including
the gateway call, rolls every DB write for this invocation back together, and only the
credential-issuance primitives (`GatewayAdminClient`, `GatewayCredentialLimits`,
`GATEWAY_MODEL_ALIASES_BY_RESIDENCY`, `generate_gateway_credential_alias`,
`write_gateway_credential_file`, `build_admin_client`) are reused, not that function's own
engine-per-call orchestration.

Idempotency: a tenant is looked up by exact name first (`app.operator.lookup.resolve_tenant`).
Found -- re-running `create` against it performs none of the three steps again, reporting each as
already in place; an existing tenant with a different residency or a non-pooled isolation tier is
a hard error rather than a silent overwrite. Not found -- all three steps run for a fresh tenant
id. The identity upsert (`ON CONFLICT (issuer, subject)`) is idempotent on its own, matching the
retired `scripts/seed.py add-membership` pattern; a fresh admin membership is only inserted if one
does not already exist for that (tenant, identity) pair.

Residency and an optional per-tenant model override are validated against
`app.config.RESIDENCY_ALLOW_LIST`/`RESIDENCY_MODEL_ALLOW_LIST` before any write happens at all --
an unrecognized selection never leaves the control plane, a secret file, or a membership
half-written.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.config import RESIDENCY_ALLOW_LIST, RESIDENCY_MODEL_ALLOW_LIST, Settings, get_settings
from app.gateway_provisioning import (
    GATEWAY_MODEL_ALIASES_BY_RESIDENCY,
    GatewayAdminClient,
    GatewayCredentialLimits,
    GatewayProvisioningError,
    build_admin_client,
    generate_gateway_credential_alias,
    write_gateway_credential_file,
)
from app.operator.lookup import TenantNotFoundError, resolve_tenant
from app.tenant_settings import TenantSettings


class UnrecognizedResidencyError(ValueError):
    """`residency` is not a key of `RESIDENCY_ALLOW_LIST` -- rejected before any write."""


class UnrecognizedModelError(ValueError):
    """`model` is not in the model allow-list for the requested residency -- rejected before any
    write."""


class TenantConflictError(ValueError):
    """An existing tenant of this name cannot be reconciled with the requested `create` call --
    a different residency, or an isolation tier this command does not create (dedicated, #71)."""


@dataclass(frozen=True, slots=True)
class CreateTenantResult:
    tenant_id: UUID
    identity_id: UUID
    control_plane: str  # "created" | "already exists"
    admin_membership: str  # "created" | "already exists"
    gateway_credential: str  # "provisioned" | "already provisioned"
    gateway_credential_alias: str


def _validate_residency(residency: str) -> None:
    if residency not in RESIDENCY_ALLOW_LIST:
        raise UnrecognizedResidencyError(
            f"unrecognized residency {residency!r}; known residencies: "
            f"{sorted(RESIDENCY_ALLOW_LIST)} (see RESIDENCY_ALLOW_LIST in app/config.py)"
        )


def _validate_model(model: str | None, residency: str) -> None:
    if model is None:
        return
    allowed = RESIDENCY_MODEL_ALLOW_LIST.get(residency, ())
    if model not in allowed:
        raise UnrecognizedModelError(
            f"unrecognized model {model!r} for residency {residency!r}; allowed models: "
            f"{sorted(allowed)} (see RESIDENCY_MODEL_ALLOW_LIST in app/config.py)"
        )


async def create_tenant(
    conn: AsyncConnection,
    *,
    tenant_name: str,
    residency: str,
    admin_email: str,
    model: str | None = None,
    issuer: str = "dev-seed",
    subject: str | None = None,
    settings: Settings | None = None,
    admin_client: GatewayAdminClient | None = None,
) -> CreateTenantResult:
    """Provision (or reconcile) a pooled tenant named `tenant_name`. See module docstring for the
    full contract. `admin_client` is the test seam (an `httpx.MockTransport`-backed
    `GatewayAdminClient` or a hand-written fake); left unset, this builds a real one from
    `Settings`, exactly as the retired seed script did.
    """
    # Validated before any write, in this order, per acceptance criteria: an unrecognized
    # residency or model-allow-list selection must reject before the control-plane record, the
    # credential, or the membership is touched.
    _validate_residency(residency)
    _validate_model(model, residency)
    tenant_settings = TenantSettings(model=model) if model is not None else None

    settings = settings or get_settings()
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
                        "SELECT residency, isolation_tier FROM control.tenants "
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
        if row["isolation_tier"] != "pooled":
            raise TenantConflictError(
                f"tenant {tenant_name!r} already exists with isolation tier "
                f"{row['isolation_tier']!r}; this command only creates pooled tenants"
            )
        control_plane_outcome = "already exists"
    else:
        tenant_id = uuid.uuid4()
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
            {
                "id": tenant_id,
                "name": tenant_name,
                "settings": json.dumps(
                    tenant_settings.model_dump(exclude_none=True) if tenant_settings else {}
                ),
            },
        )
        await conn.execute(
            text("INSERT INTO control.tenants (tenant_id, residency) VALUES (:tid, :residency)"),
            {"tid": tenant_id, "residency": residency},
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
    )
