"""The tenant-lookup helper (Spec 9 / #68, #114): resolves a tenant by id or by an unambiguous
name, shared by every operator command (`create`, `suspend`/`unsuspend`, `erase`, `list`).

Reads through `app.repositories.control.ControlRepository.enumerate_tenants` -- the same narrow,
`current_user`-gated, `SECURITY DEFINER` cross-tenant read (`control.enumerate_tenants()`,
migration 0012/0024) the tenant listing uses -- and filters its result by id or name here in
Python, rather than adding a second, targeted-query enumeration mechanism just for lookup: this
repository method is the only one granted for this cross-tenant read, and lookup and listing are
both small enough (an operator's own tenant count) that fetching every tenant to find one costs
nothing a second SQL string would have saved. Neither `public.tenants.name` nor `control.tenants`
enforces uniqueness on name, so an ambiguous name is a real possibility this helper must catch,
not a theoretical one a constraint already rules out.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncConnection

from app.repositories.control import ControlRepository


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

    tenants = await ControlRepository().enumerate_tenants(conn)

    if tenant_id is not None:
        match = next((t for t in tenants if t.tenant_id == tenant_id), None)
        if match is None:
            raise TenantNotFoundError(f"no tenant with id {tenant_id}")
        return TenantRef(tenant_id=match.tenant_id, name=match.name)

    matches = [t for t in tenants if t.name == identifier]
    if not matches:
        raise TenantNotFoundError(f"no tenant named {identifier!r}")
    if len(matches) > 1:
        raise AmbiguousTenantNameError(
            f"{len(matches)} tenants are named {identifier!r}; use the tenant id instead"
        )
    return TenantRef(tenant_id=matches[0].tenant_id, name=matches[0].name)
