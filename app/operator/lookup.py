"""The tenant-lookup helper (Spec 9 / #68): resolves a tenant by id or by an unambiguous name,
shared by every operator command (`list` today; `suspend`/`erase` in later Spec 9 tickets).

Reads `control.tenant_directory` (migration 0012) -- the one control-plane table that lists
every tenant's id and name without being bound by `control.tenants`'/`public.tenants`' per-tenant
`FORCE ROW LEVEL SECURITY` policy (see that migration's docstring for why a role-scoped bypass
policy on either table is unsafe instead). Neither `public.tenants.name` nor this table enforces
uniqueness on `name`, so an ambiguous name is a real possibility this helper must catch, not a
theoretical one a constraint already rules out.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


class TenantNotFoundError(LookupError):
    """No tenant matches the given id or name."""


class AmbiguousTenantNameError(LookupError):
    """More than one tenant shares the given name; the caller must use the id instead."""


@dataclass(frozen=True, slots=True)
class TenantRef:
    tenant_id: UUID
    name: str


async def resolve_tenant(conn: AsyncConnection, identifier: str) -> TenantRef:
    """Resolves `identifier` as a tenant id first (if it parses as a UUID), then as a name.

    Raises `TenantNotFoundError` if nothing matches, or `AmbiguousTenantNameError` if more than
    one tenant is named `identifier` -- never silently picks one.
    """
    try:
        tenant_id = UUID(identifier)
    except ValueError:
        tenant_id = None

    if tenant_id is not None:
        row = (
            await conn.execute(
                text("SELECT tenant_id, name FROM control.tenant_directory WHERE tenant_id = :id"),
                {"id": str(tenant_id)},
            )
        ).one_or_none()
        if row is None:
            raise TenantNotFoundError(f"no tenant with id {tenant_id}")
        return TenantRef(tenant_id=row.tenant_id, name=row.name)

    rows = (
        await conn.execute(
            text("SELECT tenant_id, name FROM control.tenant_directory WHERE name = :name"),
            {"name": identifier},
        )
    ).all()
    if not rows:
        raise TenantNotFoundError(f"no tenant named {identifier!r}")
    if len(rows) > 1:
        raise AmbiguousTenantNameError(
            f"{len(rows)} tenants are named {identifier!r}; use the tenant id instead"
        )
    return TenantRef(tenant_id=rows[0].tenant_id, name=rows[0].name)
