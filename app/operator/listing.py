"""The read-only tenant listing (Spec 9 / #68): the operator tool's first real command. Lets an
operator or auditor see every tenant's isolation tier, residency, database alias, and suspension
state without a direct database query.

Enumerates `control.tenant_directory` (migration 0012) for the set of tenants, then reads each
one's facts through `control.tenant_lifecycle_info`, a narrow, single-tenant `SECURITY DEFINER`
read -- never a cross-tenant policy bypass. See that migration's docstring for why.
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
    """Every tenant in `control.tenant_directory`, with its lifecycle facts attached.

    A tenant erased between the directory read and its own lifecycle-info read (a narrow race,
    since each is its own statement) is skipped rather than raised -- an operator re-running the
    listing sees it gone, exactly as `erase` intends.
    """
    directory = (
        await conn.execute(
            text("SELECT tenant_id, name FROM control.tenant_directory ORDER BY name")
        )
    ).all()

    summaries: list[TenantSummary] = []
    for row in directory:
        info = (
            await conn.execute(
                text("SELECT * FROM control.tenant_lifecycle_info(:tid)"),
                {"tid": str(row.tenant_id)},
            )
        ).one_or_none()
        if info is None:
            continue
        summaries.append(
            TenantSummary(
                tenant_id=row.tenant_id,
                name=info.name,
                isolation_tier=info.isolation_tier,
                residency=info.residency,
                database_alias=info.database_alias,
                suspended=info.suspended,
                suspended_at=info.suspended_at.isoformat() if info.suspended_at else None,
            )
        )
    return summaries
