"""Reads a tenant's own tenant-editable settings (`public.tenants.settings`, Spec 8 / #62).

The read-side counterpart to `app.tenant_settings.TenantSettings`'s write-side validation: no
repository anywhere reads `tenants.settings` back out of the database yet (it is validated on
write only). This is the one place a caller resolves the JSONB column back into a
`TenantSettings`, so every reader gets the same validated, `extra="forbid"` shape a writer
produced -- never a raw dict pulled straight out of the row.

Takes a session from `tenant_session(ctx)`, exactly like every other repository (never
`control_session()`): `tenants` is RLS-restricted to the caller's own row by the
`tenants_self_only` policy (migration 0001), so the plain `SELECT ... WHERE id = :tid` below can
never read another tenant's settings even if `ctx.tenant_id` were somehow wrong.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.context import RequestContext
from app.tenant_settings import TenantSettings


class TenantSettingsRepository:
    async def get(self, session: AsyncSession, ctx: RequestContext) -> TenantSettings:
        """`ctx.tenant_id`'s own `tenants.settings`, validated into a `TenantSettings`.

        A tenant with no row (shouldn't happen for an authenticated caller, but never trusted to
        be impossible) or a null/empty `settings` column gets the all-defaults `TenantSettings()`
        -- never raises, since every field here is optional and content tracing must default to
        off regardless of what is or isn't recorded yet.
        """
        row = (
            await session.execute(
                text("SELECT settings FROM tenants WHERE id = :tid"),
                {"tid": str(ctx.tenant_id)},
            )
        ).first()
        if row is None or row[0] is None:
            return TenantSettings()
        return TenantSettings.model_validate(row[0])


__all__ = ["TenantSettingsRepository"]
