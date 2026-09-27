"""The `add-membership` command (issue #83, ADR-0003, ADR-0010, ADR-0012, #114): attaches an
additional membership to an already-provisioned tenant -- the documented second mode
`scripts/seed.py add-membership` used to offer before the operator tool (`create`/`suspend`/
`unsuspend`/`erase`) replaced it, and `create`'s own first-admin-membership step never covered:
`create` only ever provisions the *first* admin membership for a brand-new tenant; this command
is how a second person (or a second role for the same person) joins a tenant that already exists.

A composition over the same building blocks `app.operator.create.create_tenant` uses for its own
admin membership, not a second, parallel implementation of any of them:

- the tenant is resolved by id or unambiguous name through `app.operator.lookup.resolve_tenant`
  -- the same lookup `suspend`/`unsuspend`/`erase` use, so an unknown identifier fails exactly the
  same way (`TenantNotFoundError`/`AmbiguousTenantNameError`);
- the identity is found-or-created by `(issuer, subject)` through
  `app.repositories.control.IdentityRepository.upsert` -- the same idempotency key, and the same
  `subject` default (the email) when one is not given, that `create_tenant` uses for its own admin
  identity;
- the membership itself is written through `app.repositories.memberships.ensure_membership` for a
  pooled tenant, on this function's own owner-role connection, or through
  `app.operator.dedicated_db.ensure_dedicated_membership` for a dedicated one -- exactly the same
  tier decision `create_tenant` makes for its first admin membership (ADR-0002: a dedicated
  tenant's membership can only ever live in its own database, never the pooled one).

Two things `create_tenant`'s own membership step never had to consider, because it only ever
writes one fresh membership for a brand-new tenant:

- **Idempotent, but never a silent role change.** Re-running `add-membership` with the same
  `(tenant, issuer, subject, role)` is a no-op, reported as `"already exists"` -- but if an
  existing membership for that identity already carries a *different* role, this refuses
  (`app.repositories.memberships.MembershipRoleConflictError`) rather than changing it: changing
  an existing membership's role is a deliberately separate, out-of-scope command.
- **Refuses a suspended tenant.** CONTEXT.md's "Suspension" glossary entry and ADR-0010 describe
  suspension as a state in which nothing is provisioned for the tenant; `add-membership` reads the
  tenant's own record (`ControlRepository.get_record`, the same owner-role read `create`'s
  reconciliation and `erase`'s own suspension read use) and raises `TenantSuspendedError` before
  touching the identity or the membership if it is suspended.

`role` is validated against `app.context.ROLES` before any write, twice: once by the CLI's own
`argparse` `choices=` (`app.operator.cli.build_parser`), and again here (`UnrecognizedRoleError`)
for a caller that reaches this function directly, the same defense-in-depth
`app.operator.create._validate_isolation_tier` applies to `isolation_tier`.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncConnection

from app.context import ROLES, Role
from app.operator.dedicated_db import ensure_dedicated_database, ensure_dedicated_membership
from app.operator.lookup import resolve_tenant
from app.repositories.control import ControlRepository, IdentityRepository
from app.repositories.memberships import (
    MembershipRoleConflictError,
    ensure_membership,
    get_role_owner,
)

DEFAULT_ISSUER = "dev-seed"
"""Matches `app.operator.create.create_tenant`'s own default -- the identity a re-run of either
command names, with no `--issuer` given, is the same identity."""


class UnrecognizedRoleError(ValueError):
    """`role` is not one of `app.context.ROLES` -- rejected before any write."""


class TenantSuspendedError(RuntimeError):
    """`add-membership` refuses to run against a suspended tenant (ADR-0010, CONTEXT.md's
    "Suspension" glossary entry: nothing is provisioned for a suspended tenant) -- changes nothing
    when this is raised. Carries `tenant_id` (already resolved by the time this is raised) so
    `app.operator.cli._run` can audit the failure against the real tenant rather than the
    unscoped sentinel -- see that module's own comment."""

    def __init__(self, tenant_id: UUID, message: str) -> None:
        self.tenant_id = tenant_id
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class AddMembershipResult:
    tenant_id: UUID
    tenant_name: str
    identity_id: UUID
    role: str
    membership: str  # "created" | "already exists"

    def render(self) -> str:
        return (
            f"identity {self.identity_id} ({self.role}) in {self.tenant_name!r} "
            f"({self.tenant_id}): membership {self.membership}"
        )

    @property
    def audit_outcome(self) -> str:
        """The one line `app.operator.cli` writes to the operator-action log for this command --
        never printed to stdout (see `render()`)."""
        return f"ok: tenant {self.tenant_id} membership {self.membership} (role: {self.role})"


async def add_membership(
    conn: AsyncConnection,
    identifier: str,
    *,
    role: Role,
    email: str,
    issuer: str = DEFAULT_ISSUER,
    subject: str | None = None,
) -> AddMembershipResult:
    """Attach a membership of `role` to the tenant named or identified by `identifier`, for the
    identity matching `(issuer, subject)` (found or created). See module docstring for the full
    contract. Raises before any write for an unrecognized role, an unresolvable tenant, or a
    suspended one; raises `app.repositories.memberships.MembershipRoleConflictError` after the
    identity is resolved but before any membership write if an existing membership for that
    identity already carries a different role.
    """
    if role not in ROLES:
        raise UnrecognizedRoleError(f"unrecognized role {role!r}; known roles: {sorted(ROLES)}")

    ref = await resolve_tenant(conn, identifier)
    record = await ControlRepository().get_record(conn, ref.tenant_id)
    if record.suspended:
        raise TenantSuspendedError(
            ref.tenant_id,
            f"tenant {ref.name!r} ({ref.tenant_id}) is suspended; add-membership provisions "
            "nothing for a suspended tenant (ADR-0010)",
        )

    subject = subject or email
    identity_id = await IdentityRepository().upsert(
        conn, id=uuid4(), issuer=issuer, subject=subject, email=email
    )

    if record.isolation_tier == "pooled":
        existing_role = await get_role_owner(conn, tenant_id=ref.tenant_id, identity_id=identity_id)
        if existing_role is not None and existing_role != role:
            raise MembershipRoleConflictError(ref.tenant_id, identity_id, existing_role, role)
        outcome = await ensure_membership(
            conn, tenant_id=ref.tenant_id, identity_id=identity_id, role=role
        )
    else:
        assert record.database_alias is not None  # enforced by migration 0005's CHECK constraint
        dedicated = await ensure_dedicated_database(alias=record.database_alias, admin_url=None)
        # The dedicated database's `tenants` stub row already exists (written by `create`'s own
        # provisioning, `ON CONFLICT (id) DO NOTHING`), so an empty settings payload here changes
        # nothing on the row that matters -- no need to reconstruct it from the control-plane
        # record.
        outcome = await ensure_dedicated_membership(
            owner_dsn=dedicated.owner_dsn,
            tenant_id=ref.tenant_id,
            tenant_name=ref.name,
            tenant_settings_json="{}",
            identity_id=identity_id,
            issuer=issuer,
            subject=subject,
            email=email,
            role=role,
        )

    return AddMembershipResult(
        tenant_id=ref.tenant_id,
        tenant_name=ref.name,
        identity_id=identity_id,
        role=role,
        membership=outcome,
    )
