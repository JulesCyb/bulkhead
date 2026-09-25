"""SQLAlchemy models. Every domain table has tenant_id + RLS (migration 0001_initial.py).

New table? Four mandatory parts in the migration: tenant_id NOT NULL REFERENCES tenants(id),
an index on tenant_id, ENABLE + FORCE ROW LEVEL SECURITY, a policy with USING and WITH CHECK.

An embedded-Postgres integration test (`tests/test_rls_integration.py`,
`test_public_schema_tenant_isolation_invariant`) walks every table actually present in the
`public` schema and fails the build if one of them is missing forced RLS or a policy that both
restricts (USING) and validates (WITH CHECK) against `app.tenant_id` — unless that table is
named in `TENANT_ISOLATION_EXCEPTIONS` below. A legitimately tenant-less table is added to that
one set, never to a second, separate list that could drift from what the test enforces.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIMENSIONS = 1536

# Tables in the public schema exempt from the tenant-isolation invariant test. Only the
# bookkeeping table Alembic itself creates (`alembic_version`, tracking which migrations have
# run) belongs here: it holds no tenant data, is not created by any migration in this repo, and
# exists before the first tenant does. Adding a genuinely tenant-less table to the schema means
# adding it here — nowhere else — so the test and the checklist above can never drift apart.
TENANT_ISOLATION_EXCEPTIONS: frozenset[str] = frozenset({"alembic_version"})


class Base(DeclarativeBase):
    pass


# Not an ORM-mapped class: control.identities (migration 0003, ADR-0003) is reached only through
# app/repositories/control.py's raw SQL against the narrow control.identity_lookup view -- app
# never gets a grant on the table itself (see migration 0010's docstring). This bare Table exists
# solely so SQLAlchemy's metadata can resolve the cross-schema foreign keys on Document below; it
# is never queried, written to, or migrated from here.
control_identities = Table(
    "identities",
    Base.metadata,
    Column("id", PGUUID(as_uuid=True), primary_key=True),
    schema="control",
)


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200))
    # Tenant-specific configuration: model choice, prompts, limits, feature flags.
    settings: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Membership(Base):
    """A tenant's roster of who belongs to it (ADR-0003, Spec 2 / #23): tenant, a cross-schema
    reference to the global `control.identities` row, and that identity's role in the tenant.
    Replaces the retired `users` table -- an identity is global, a membership is per tenant."""

    __tablename__ = "memberships"
    # Mirrors migration 0009 exactly (constraint + index names included), so that a later
    # `alembic revision --autogenerate` does not emit destructive drift.
    __table_args__ = (
        UniqueConstraint("tenant_id", "identity_id"),
        Index("memberships_tenant_idx", "tenant_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[UUID] = mapped_column(ForeignKey("tenants.id"))
    # control.identities lives outside the public schema and carries no RLS policy of its own
    # (ADR-0003) -- this is a plain FK, not a tenant-isolated relation.
    identity_id: Mapped[UUID] = mapped_column(ForeignKey("control.identities.id"))
    role: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (Index("documents_tenant_idx", "tenant_id"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[UUID] = mapped_column(ForeignKey("tenants.id"))
    title: Mapped[str] = mapped_column(String(500))
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS), nullable=True
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # created_by/updated_by/updated_at (migration 0010, #29): the audit-column pattern later
    # tenant tables copy verbatim. Both reference control.identities, never a membership -- see
    # that migration's docstring. `server_default` here must match the migration's column
    # DEFAULT exactly: it is what tells SQLAlchemy these are DB-computed values to omit from the
    # INSERT and fetch back via RETURNING, rather than sending an explicit NULL for an attribute
    # the application never set (which would violate the NOT NULL constraint instead of letting
    # the database's own default fire). updated_by is refreshed again on every UPDATE by the
    # trigger created_by/updated_by (migration 0010) — the ORM's server_default only governs the
    # value at INSERT time.
    created_by: Mapped[UUID] = mapped_column(
        ForeignKey("control.identities.id"),
        server_default=text("current_setting('app.identity_id', true)::uuid"),
    )
    updated_by: Mapped[UUID] = mapped_column(
        ForeignKey("control.identities.id"),
        server_default=text("current_setting('app.identity_id', true)::uuid"),
    )
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
