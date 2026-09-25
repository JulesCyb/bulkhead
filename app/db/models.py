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
from sqlalchemy import DateTime, ForeignKey, Index, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
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


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200))
    # Tenant-specific configuration: model choice, prompts, limits, feature flags.
    settings: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class User(Base):
    __tablename__ = "users"
    # Mirrors migration 0001 exactly (constraint + index names included), so that a later
    # `alembic revision --autogenerate` does not emit destructive drift.
    __table_args__ = (
        UniqueConstraint("tenant_id", "email"),
        Index("users_tenant_idx", "tenant_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[UUID] = mapped_column(ForeignKey("tenants.id"))
    email: Mapped[str] = mapped_column(String(320))
    role: Mapped[str] = mapped_column(String(50), default="member")
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
