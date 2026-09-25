"""The `suspend`/`unsuspend` commands (Spec 9 / #69, ADR-0010): resolve a tenant by id or
unambiguous name (`app.operator.lookup`), then flip its suspension state through
`control.set_tenant_suspended()` (migration 0024) -- the one `SECURITY DEFINER` write path the
operator role is granted onto `control.tenants`.

Idempotent by construction: re-running `suspend` against an already-suspended tenant, or
`unsuspend` against an already-active one, is reported as a no-op (`changed=False`), never an
error -- the caller (`app.operator.cli`) turns that into the outcome string the audit log records.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.operator.lookup import resolve_tenant


@dataclass(frozen=True, slots=True)
class SuspendResult:
    tenant_id: UUID
    name: str
    suspended: bool
    changed: bool
    suspended_at: datetime | None


async def set_tenant_suspended(
    conn: AsyncConnection, identifier: str, *, suspended: bool
) -> SuspendResult:
    """Resolves `identifier` (id or unambiguous name) and sets its suspension state to
    `suspended`. Raises `app.operator.lookup.TenantNotFoundError`/`AmbiguousTenantNameError` if
    `identifier` does not resolve to exactly one tenant -- the same lookup every other command
    uses, never a second, looser resolution just for this one.
    """
    ref = await resolve_tenant(conn, identifier)
    row = (
        await conn.execute(
            text(
                "SELECT changed, suspended_at FROM control.set_tenant_suspended(:tid, :suspended)"
            ),
            {"tid": str(ref.tenant_id), "suspended": suspended},
        )
    ).one()
    return SuspendResult(
        tenant_id=ref.tenant_id,
        name=ref.name,
        suspended=suspended,
        changed=row.changed,
        suspended_at=row.suspended_at,
    )
