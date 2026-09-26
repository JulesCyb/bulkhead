"""The read-only tenant listing (Spec 9 / #68, #114): the operator tool's first real command. Lets
an operator or auditor see every tenant's isolation tier, residency, database alias, and
suspension state without a direct database query.

Reads through `app.repositories.control.ControlRepository.enumerate_tenants`, which calls
`control.enumerate_tenants()` (migration 0012/0024): a narrow, `current_user`-gated, `SECURITY
DEFINER` cross-tenant read -- never a plain role-scoped bypass policy on `control.tenants` or
`public.tenants` directly, which (see that migration's docstring) would leak through
`control.tenants_view` to the `app` role. Not granted to `app`; only `app_owner` may call it.
`app.operator.lookup.resolve_tenant` reads the exact same enumeration to resolve one tenant by id
or name, rather than this module and that one issuing two different queries against the same
function.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncConnection

from app.repositories.control import ControlRepository, TenantSummary

__all__ = ["TenantSummary", "list_tenants"]


async def list_tenants(conn: AsyncConnection) -> list[TenantSummary]:
    """Every tenant in the control plane, with its lifecycle facts."""
    return await ControlRepository().enumerate_tenants(conn)
