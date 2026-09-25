"""The read-only tenant listing (Spec 9 / #68): the operator tool's first real command. Lets an
operator or auditor see every tenant's isolation tier, residency, database alias, and suspension
state without a direct database query.

Reads `control.enumerate_tenants()` (migration 0012): a narrow, current_user-gated, `SECURITY
DEFINER` cross-tenant read -- never a plain role-scoped bypass policy on `control.tenants` or
`public.tenants` directly, which (see that migration's docstring) would leak through
`control.tenants_view` to the `app` role. Not granted to `app`; only `app_owner` may call it.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


@dataclass(frozen=True, slots=True)
class TenantSummary:
    tenant_id: UUID
    name: str
    isolation_tier: str
    residency: str | None
    database_alias: str | None
    suspended: bool
    suspended_at: str | None


async def list_tenants(conn: AsyncConnection) -> list[TenantSummary]:
    """Every tenant in the control plane, with its lifecycle facts."""
    rows = (
        await conn.execute(text("SELECT * FROM control.enumerate_tenants() ORDER BY name"))
    ).all()
    return [
        TenantSummary(
            tenant_id=row.tenant_id,
            name=row.name,
            isolation_tier=row.isolation_tier,
            residency=row.residency,
            database_alias=row.database_alias,
            suspended=row.suspended,
            suspended_at=row.suspended_at.isoformat() if row.suspended_at else None,
        )
        for row in rows
    ]
