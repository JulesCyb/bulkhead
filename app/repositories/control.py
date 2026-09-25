"""Control-plane data access: the two narrow, cross-tenant reads the `app` role is granted
(ADR-0003 / ADR-0011 / issue #22). Both take a session from `control_session()` (app/db/session.py)
-- never `tenant_session(ctx)`, and never any other query against `control.*`.

The application never creates, changes, or removes an identity; only the owner-role seed/admin
path does that. `find_by_issuer_and_subject` and `get` below are read-only by construction: they
issue a single SELECT each and return None on no match rather than raising.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


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
