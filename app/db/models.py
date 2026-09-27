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
    BigInteger,
    Column,
    DateTime,
    FetchedValue,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
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


class AgentCredential(Base):
    """A credential issued for an agent identity's automation (ADR-0005, Spec 6 / #45).

    `identity_id` is the *agent* identity the credential authenticates as; `created_by` is the
    (typically human, admin-role) identity that issued it -- the same actor/means split
    documents' created_by/updated_by draws, applied here to who-issued vs. who-it-is-for instead
    of who-wrote vs. who-last-touched. `public_id` and `secret_hash` are deliberately separate
    columns: verification is a direct lookup by `public_id` (UNIQUE with tenant_id, indexed),
    never a scan comparing a presented secret against every hash a tenant has issued."""

    __tablename__ = "agent_credentials"
    __table_args__ = (
        UniqueConstraint("tenant_id", "public_id"),
        Index("agent_credentials_tenant_idx", "tenant_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[UUID] = mapped_column(ForeignKey("tenants.id"))
    identity_id: Mapped[UUID] = mapped_column(ForeignKey("control.identities.id"))
    name: Mapped[str] = mapped_column(String(200))
    public_id: Mapped[str] = mapped_column(String(64))
    secret_hash: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_by: Mapped[UUID] = mapped_column(
        ForeignKey("control.identities.id"),
        server_default=text("current_setting('app.identity_id', true)::uuid"),
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


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


class Conversation(Base):
    """Server-side chat history (ADR-0006, Spec 4 / #32): keyed by the pair of tenant and the
    client's own conversation id (the Vercel chat `id`), not a server-generated one -- the client
    names a conversation and the server recognizes it on the next request. `created_by` is the
    audit-column pattern from migration 0010, applied here for the first time to a table that
    isn't `documents`; there is no `updated_by`/trigger pair -- `last_activity_at` is refreshed by
    a `SECURITY DEFINER` trigger on `messages` (migration 0020), never by the application issuing
    an `UPDATE` it has no grant to make."""

    __tablename__ = "conversations"
    __table_args__ = (Index("conversations_tenant_idx", "tenant_id"),)

    tenant_id: Mapped[UUID] = mapped_column(ForeignKey("tenants.id"), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    created_by: Mapped[UUID] = mapped_column(
        ForeignKey("control.identities.id"),
        server_default=text("current_setting('app.identity_id', true)::uuid"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Message(Base):
    """One library-native `pydantic_ai.messages` message (a `ModelRequest` or `ModelResponse`),
    serialized with the library's own adapter -- never a hand-rolled shape (Spec 4, story 21).
    `sequence` is a monotonically increasing, application-assigned counter within its
    conversation; reloading orders by it, not by `created_at`, because timestamp ties are not a
    safe substitute for the exact order a run produced (Spec 4, stories 20/21)."""

    __tablename__ = "messages"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "conversation_id"],
            ["conversations.tenant_id", "conversations.conversation_id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("tenant_id", "conversation_id", "sequence"),
        Index("messages_tenant_idx", "tenant_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[UUID] = mapped_column(ForeignKey("tenants.id"))
    conversation_id: Mapped[str] = mapped_column(String(200))
    sequence: Mapped[int] = mapped_column(Integer)
    payload: Mapped[dict] = mapped_column(JSONB)
    created_by: Mapped[UUID] = mapped_column(
        ForeignKey("control.identities.id"),
        server_default=text("current_setting('app.identity_id', true)::uuid"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PendingAction(Base):
    """A writing-tool call awaiting or having received an approval (ADR-0007, Spec 5 / #37):
    tenant, conversation, tool, a hash of the exact arguments, the asking membership, and an
    expiry -- written down *before* the approval is ever shown to the member, so verification
    always has a trustworthy record to check against, never the client's own message.

    `asking_membership_id`/`resolved_by` reference `memberships`, not `control.identities`: a
    pending action is a tenant-scoped fact about a *membership*'s role, unlike the
    `created_by`/`updated_by` audit columns elsewhere in this schema, which name a global
    identity (migration 0010's docstring)."""

    __tablename__ = "pending_actions"
    __table_args__ = (Index("pending_actions_tenant_idx", "tenant_id"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[UUID] = mapped_column(ForeignKey("tenants.id"))
    conversation_id: Mapped[str] = mapped_column(String(200))
    tool_name: Mapped[str] = mapped_column(String(200))
    args_hash: Mapped[str] = mapped_column(String(64))
    # The model's own tool call id (migration 0038, #40): the key pydantic-ai's own approval
    # resolution uses across the propose/resume runs -- see that migration's docstring.
    tool_call_id: Mapped[str] = mapped_column(String(200))
    asking_membership_id: Mapped[UUID] = mapped_column(ForeignKey("memberships.id"))
    status: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_by: Mapped[UUID | None] = mapped_column(ForeignKey("memberships.id"), nullable=True)


class StandingGrant(Base):
    """A tenant admin's standing authorization for one agent identity to call one writing tool
    with no person present (ADR-0005, ADR-0007, Spec 5 / #38).

    `agent_membership_id`/`granted_by`/`revoked_by` all reference `memberships`, not
    `control.identities` -- the same split `PendingAction` draws and for the same reason: a grant
    is a tenant-scoped fact about a membership's role, not about the global identity.

    Uniqueness ("at most one active grant per tenant, agent identity, and tool") is enforced by
    the database, not here: `standing_grants_active_uidx` (migration 0031) is a partial unique
    index on `(agent_membership_id, tool_name) WHERE revoked_at IS NULL` -- a revoked grant never
    counts against it, so revoking and re-granting the same membership/tool pair is always legal.
    """

    __tablename__ = "standing_grants"
    __table_args__ = (Index("standing_grants_tenant_idx", "tenant_id"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[UUID] = mapped_column(ForeignKey("tenants.id"))
    agent_membership_id: Mapped[UUID] = mapped_column(ForeignKey("memberships.id"))
    tool_name: Mapped[str] = mapped_column(String(200))
    granted_by: Mapped[UUID] = mapped_column(ForeignKey("memberships.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_by: Mapped[UUID | None] = mapped_column(ForeignKey("memberships.id"), nullable=True)


class ApprovalAuditEvent(Base):
    """One append-only milestone of the approval mechanism (ADR-0007, Spec 5 / #39): a write
    requested, approved, refused, expired, denied for lack of a grant, executed, or failed to
    execute. Distinct from the `created_by`/`updated_by` audit-column pattern 0010 established
    for `documents` -- that answers "who wrote this row"; this answers "who authorized this
    write, and how" (CLAUDE.md rule 4).

    `actor_membership_id` references `memberships`, not `control.identities` -- the same split
    `PendingAction` and `StandingGrant` already draw: the record is a tenant-scoped fact about a
    membership's role at the moment the milestone happened.

    `pending_action_id`/`standing_grant_id` are plain, unconstrained `uuid` columns (no FK),
    deliberately: this row must document, and outlive, the pending action or standing grant it
    names, the same way `control.operator_actions`/`control.tenant_erasures` (migration 0004)
    outlive the tenant they document. At most one of the two is set for most kinds; a
    `denied_for_lack_of_grant` event -- by definition, no grant exists -- leaves both null.

    `seq` is a database-generated identity column used only for ordering (see migration 0035's
    docstring for why `created_at` alone cannot be trusted to order events written in the same
    transaction).

    `means_kind`/`means_id` (migration 0042, #117) are a second, distinct fact from
    `pending_action_id`/`standing_grant_id` above: those name the *approval* means (which pending
    action or standing grant authorized the write); these name the *delegation* means (ADR-0005) --
    whether the request was a person acting through the assistant or an agent identity acting on
    its own credential. Both nullable: a context with no means attached (a test, the `stdio`
    development fallback) writes both null.
    """

    __tablename__ = "approval_audit_events"
    __table_args__ = (
        Index("approval_audit_events_tenant_idx", "tenant_id"),
        Index("approval_audit_events_pending_action_idx", "pending_action_id"),
        Index("approval_audit_events_standing_grant_idx", "standing_grant_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    # GENERATED ALWAYS AS IDENTITY (migration 0035) -- never set from Python; FetchedValue keeps
    # this column out of the INSERT statement entirely, exactly as Postgres's identity clause
    # requires (it errors on any explicit value, even NULL, without OVERRIDING SYSTEM VALUE).
    seq: Mapped[int] = mapped_column(BigInteger, server_default=FetchedValue())
    tenant_id: Mapped[UUID] = mapped_column(ForeignKey("tenants.id"))
    kind: Mapped[str] = mapped_column(String(30))
    tool_name: Mapped[str] = mapped_column(String(200))
    actor_membership_id: Mapped[UUID] = mapped_column(ForeignKey("memberships.id"))
    pending_action_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    standing_grant_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    # The delegation means (ADR-0005, migration 0042, #117) -- distinct from the approval means
    # (pending_action_id/standing_grant_id) above. Null when the writing context carried none.
    means_kind: Mapped[str | None] = mapped_column(String(20), nullable=True)
    means_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    details: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
