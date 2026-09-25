"""Read-only access to the control plane (Spec 7 / #52, ADR-0011).

Unlike `app/repositories/documents.py`, there is no write path here on purpose: `control.*` is
owned by `app_owner` and written only by the owning role (migrations, the operator tool); the
application reaches it exclusively through `control.tenants_view`, a `security_invoker` view
granted `SELECT` only. The session still comes from `tenant_session(ctx)`, so RLS filters the
view to the caller's own tenant row regardless of what this repository asks for.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.context import RequestContext


class ControlRepository:
    async def get_gateway_credential_alias(
        self, session: AsyncSession, ctx: RequestContext
    ) -> str | None:
        """The alias of `ctx.tenant_id`'s gateway credential, or None if none is recorded yet."""
        row = (
            await session.execute(
                text(
                    "SELECT gateway_credential_alias FROM control.tenants_view "
                    "WHERE tenant_id = :tid"
                ),
                {"tid": str(ctx.tenant_id)},
            )
        ).first()
        return row[0] if row else None
