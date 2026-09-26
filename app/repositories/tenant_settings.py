"""Reads a tenant's own tenant-editable settings (`public.tenants.settings`, Spec 8 / #62).

The read-side counterpart to `app.tenant_settings.TenantSettings`'s write-side validation: the
one place the JSONB column is resolved back into a `TenantSettings`, so every reader gets the same
validated, `extra="forbid"` shape a writer produced -- never a raw dict pulled straight out of the
row. Its one caller is the tenant-record read (`ControlRepository.get_tenant_record`, #104): model,
tracing, and retention all read the settings from the record it builds (#105), never from here.

Takes a session with `app.tenant_id` set (`tenant_record_session(tenant_id)`), never
`control_session()`: `tenants` is RLS-restricted to the caller's own row by the
`tenants_self_only` policy (migration 0001), so the plain `SELECT ... WHERE id = :tid` below can
never read another tenant's settings even if `tenant_id` were somehow wrong.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.tenant_settings import TenantSettings


class TenantSettingsRepository:
    async def get_for_tenant(self, session: AsyncSession, *, tenant_id: UUID) -> TenantSettings:
        """`tenant_id`'s own `tenants.settings`, validated into a `TenantSettings`, for the
        tenant-record read (`app.repositories.control.ControlRepository.get_tenant_record`,
        #104), whose session has `app.tenant_id` set to `tenant_id`.

        A tenant with no row (shouldn't happen for an authenticated caller, but never trusted to
        be impossible) or a null/empty `settings` column gets the all-defaults `TenantSettings()`
        -- never raises, since every field here is optional and content tracing must default to
        off regardless of what is or isn't recorded yet.
        """
        row = (
            await session.execute(
                text("SELECT settings FROM tenants WHERE id = :tid"),
                {"tid": str(tenant_id)},
            )
        ).first()
        if row is None or row[0] is None:
            return TenantSettings()
        return TenantSettings.model_validate(row[0])


__all__ = ["TenantSettingsRepository"]
