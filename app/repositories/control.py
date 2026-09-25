"""Control-plane data access: the two narrow, cross-tenant reads the `app` role is granted
(ADR-0003 / ADR-0011 / issue #22). Both take a session from `control_session()` (app/db/session.py)
-- never `tenant_session(ctx)`, and never any other query against `control.*`.

The application never creates, changes, or removes an identity; only the owner-role seed/admin
path does that. `find_by_issuer_and_subject` and `get` below are read-only by construction: they
issue a single SELECT each and return None on no match rather than raising.

`ControlRepository` (Spec 7 / #52) is the one tenant-scoped read: it takes a session from
`tenant_session(ctx)` and reads the caller's own row through `control.tenants_view`, which RLS
filters to `ctx.tenant_id`.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.context import RequestContext


class Identity(BaseModel):
    """Exactly what `control.identity_lookup` exposes -- never `display_name`/`email`, even
    though `control.identities` carries those columns."""

    id: UUID
    issuer: str
    subject: str


class TenantAuthSettings(BaseModel):
    issuer: str | None
    suspended: bool


class IdentityRepository:
    async def find_by_issuer_and_subject(
        self, session: AsyncSession, *, issuer: str, subject: str
    ) -> Identity | None:
        row = (
            (
                await session.execute(
                    text(
                        "SELECT id, issuer, subject FROM control.identity_lookup "
                        "WHERE issuer = :issuer AND subject = :subject"
                    ),
                    {"issuer": issuer, "subject": subject},
                )
            )
            .mappings()
            .one_or_none()
        )
        return Identity.model_validate(dict(row)) if row is not None else None


class TenantAuthSettingsRepository:
    """Reads a named tenant's configured issuer and suspension state through
    control.tenant_auth_settings() -- the only narrow read permitted for a *specific* tenant
    over a control-plane session that otherwise sets no tenant context at all."""

    async def get(
        self, session: AsyncSession, *, tenant_id: UUID, default_issuer: str | None = None
    ) -> TenantAuthSettings | None:
        row = (
            (
                await session.execute(
                    text(
                        "SELECT identity_issuer, suspended_at "
                        "FROM control.tenant_auth_settings(:tenant_id)"
                    ),
                    {"tenant_id": str(tenant_id)},
                )
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            return None
        return TenantAuthSettings(
            issuer=row["identity_issuer"] or default_issuer,
            suspended=row["suspended_at"] is not None,
        )


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

    async def get_residency(self, session: AsyncSession, ctx: RequestContext) -> str | None:
        """`ctx.tenant_id`'s residency (ADR-0008), or None if the tenant has no control-plane
        row at all -- the caller falls back to the deployment's own `Settings.residency` in that
        case, same as `resolve_gateway_credential_alias` does for a missing alias."""
        row = (
            await session.execute(
                text("SELECT residency FROM control.tenants_view WHERE tenant_id = :tid"),
                {"tid": str(ctx.tenant_id)},
            )
        ).first()
        return row[0] if row else None
