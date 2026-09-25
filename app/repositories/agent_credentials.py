"""Agent-credential repository (ADR-0005, Spec 6 / #45): the only path to `agent_credentials`.

The session comes from `tenant_session(ctx)` and is therefore tenant-bound; RLS filters every
method to `ctx.tenant_id` before any code here runs (a credential created under one tenant is
simply not a row a second tenant's session can see, by construction -- not by a WHERE clause
this repository would otherwise have to get right).

**`create()`'s own invariant (finding from the 2026-09-25 review of #46): a credential must name
an agent identity in the caller's own tenant.** `ctx.require_role("admin")` (checked by
`app/tools/agent_identities.py`, before this is ever reached) only says the *caller* may issue
credentials; it says nothing about whether `identity_id` is an agent identity, or whether it
belongs to this tenant at all. Nothing before this change checked either, so an admin of tenant A
could mint a credential row naming tenant B's agent identity, or a person's identity in their own
tenant -- unusable later (the minted token would never resolve to anything a client could use),
but it still lets a caller learn whether an arbitrary identity id exists at all, and it still
writes a row into `agent_credentials` that violates the "credential belongs to an agent identity
of this tenant" invariant ADR-0005 describes, `agent_credentials.identity_id`'s FK to
`control.identities` be damned (that FK only checks the row exists *somewhere*, not that it is an
agent, and not that it is a member here).

`create()` re-reads `identity_id`'s membership role in `ctx.tenant_id`, fresh, the same way
`StandingGrantRepository.create()` (`app/repositories/standing_grants.py`) already re-reads a
target membership's role before trusting it, and refuses (`UnknownAgentIdentity`) unless it is
exactly `'agent'`. This is a membership check, not a `control.identities.kind` check, because kind
is not readable through any narrow path `app` is granted (`app/repositories/control.py` exposes
only `issuer`/`subject`, never `kind`) -- but it does not need to be: the only way anything in this
codebase ever creates a membership with role `'agent'` is `control.create_agent_identity`
(migration 0032), and that function always pairs it with an identity it inserts as `kind =
'agent'` in the same statement. There is no path -- no tool, no route -- that grants an `'agent'`
role membership to any other identity. So "has an active `'agent'`-role membership in this
tenant" and "is an agent identity that belongs to this tenant" are the same fact today; if a
future change ever lets an admin hand an ordinary membership the `agent` role directly, revisit
this and check `kind` too.

The check is a single query, scoped by `ctx.tenant_id` exactly like every other read here, so an
identity that does not exist at all, one that belongs to another tenant, and one that exists here
but is a person -- not an agent -- all resolve to the *same* `None`/wrong-role outcome and the
same `UnknownAgentIdentity`. The caller (`app/tools/agent_identities.py`) turns that into the same
404 an unknown id would get; there is no way to tell the three cases apart from the response.

**Hashing choice.** A credential's secret is not a human-chosen password: it is generated here,
once, from `secrets.token_urlsafe` (a CSPRNG), with 256 bits of entropy. The slow, salted KDFs
password storage needs (argon2, scrypt, bcrypt) exist to blunt an offline dictionary/brute-force
attack against *low-entropy, human-chosen* secrets -- they buy time by making each guess
expensive. A 256-bit random token has no dictionary to attack: even at billions of hashes per
second, exhausting the keyspace is not a feasible attack, so a slow KDF buys nothing here beyond
extra CPU on every verification. A single, unsalted SHA-256 digest of the secret is what GitHub,
Stripe, and Postgres' own high-entropy API-token schemes use for exactly this reason. Salting is
pointless against a token that never repeats and needs no dictionary defeated; a hash is stored
only so a database compromise cannot yield a value an attacker could present again -- SHA-256
already gives that.

Comparison uses `hmac.compare_digest` (constant-time), so a timing side-channel over repeated
verification attempts cannot narrow down a correct hash byte by byte.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.context import RequestContext
from app.db.models import AgentCredential
from app.repositories.memberships import MembershipRepository


def _hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


class UnknownAgentIdentity(ValueError):
    """Raised by `create()` when `identity_id` does not carry an active `agent`-role membership
    in `ctx.tenant_id` -- including an identity this tenant has no record of at all, one that
    belongs to another tenant, and one that exists here but is a person, not an agent. RLS and
    this repository's own tenant-scoped query make all three indistinguishable by construction;
    the caller must answer all three the same way (a plain "not found"), never revealing which
    one actually happened."""


class IssuedCredential(BaseModel):
    """Returned exactly once, at creation -- the only moment the plaintext secret exists outside
    the caller's own generation of it. Never reconstructable and never returned again."""

    id: UUID
    identity_id: UUID
    name: str
    public_id: str
    secret: str
    created_at: datetime


class CredentialRecord(BaseModel):
    """What `list_for_tenant` returns: metadata only, never a secret or its hash."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    identity_id: UUID
    name: str
    public_id: str
    created_at: datetime
    revoked_at: datetime | None
    last_used_at: datetime | None


class VerifiedCredential(BaseModel):
    """What a successful `verify_and_touch` resolves to: enough to build a request context for
    the agent identity the credential authenticates as."""

    id: UUID
    identity_id: UUID
    tenant_id: UUID


class AgentCredentialRepository:
    async def create(
        self, session: AsyncSession, ctx: RequestContext, *, identity_id: UUID, name: str
    ) -> IssuedCredential:
        """Generate a new public identifier and secret, persist only the secret's hash, and
        return the plaintext secret once -- it is never stored and never retrievable again.

        Refuses outright (`UnknownAgentIdentity`, see module docstring) unless `identity_id`
        currently carries an active `agent`-role membership in `ctx.tenant_id` -- checked fresh,
        here, before anything is written."""
        role = await MembershipRepository().get_role(session, ctx, identity_id=identity_id)
        if role != "agent":
            raise UnknownAgentIdentity(
                f"identity {identity_id} has no agent membership in this tenant"
            )
        public_id = f"agt_{secrets.token_urlsafe(16)}"
        secret = secrets.token_urlsafe(32)
        credential = AgentCredential(
            tenant_id=ctx.tenant_id,
            identity_id=identity_id,
            name=name,
            public_id=public_id,
            secret_hash=_hash_secret(secret),
        )
        session.add(credential)
        await session.flush()
        return IssuedCredential(
            id=credential.id,
            identity_id=credential.identity_id,
            name=credential.name,
            public_id=credential.public_id,
            secret=secret,
            created_at=credential.created_at,
        )

    async def list_for_tenant(
        self, session: AsyncSession, ctx: RequestContext
    ) -> list[CredentialRecord]:
        """Every credential issued in `ctx.tenant_id` -- metadata only, never a secret or hash."""
        rows = (
            await session.execute(
                select(AgentCredential).where(AgentCredential.tenant_id == ctx.tenant_id)
            )
        ).scalars()
        return [CredentialRecord.model_validate(row) for row in rows]

    async def revoke(
        self, session: AsyncSession, ctx: RequestContext, *, credential_id: UUID
    ) -> bool:
        """Mark a credential dead without deleting its row. Idempotent: revoking an
        already-revoked (or unknown, or cross-tenant) credential updates nothing and returns
        False; the agent identity and any other credential issued to it are untouched -- this
        statement's WHERE names only the one credential id, nothing else."""
        result = await session.execute(
            update(AgentCredential)
            .where(
                AgentCredential.tenant_id == ctx.tenant_id,
                AgentCredential.id == credential_id,
                AgentCredential.revoked_at.is_(None),
            )
            .values(revoked_at=func.now())
        )
        return result.rowcount > 0

    async def verify_and_touch(
        self, session: AsyncSession, ctx: RequestContext, *, public_id: str, secret: str
    ) -> VerifiedCredential | None:
        """Look up `public_id` directly (indexed, `UNIQUE (tenant_id, public_id)` -- never a
        scan), then check the hash and the revoked state together. `last_used_at` advances only
        on a successful verification: an unknown identifier, a revoked credential, and a wrong
        secret all leave the row untouched, and all return None indistinguishably."""
        row = (
            await session.execute(
                select(AgentCredential).where(
                    AgentCredential.tenant_id == ctx.tenant_id,
                    AgentCredential.public_id == public_id,
                )
            )
        ).scalar_one_or_none()
        if row is None or row.revoked_at is not None:
            return None
        if not hmac.compare_digest(row.secret_hash, _hash_secret(secret)):
            return None

        await session.execute(
            update(AgentCredential)
            .where(AgentCredential.id == row.id)
            .values(last_used_at=func.now())
        )
        return VerifiedCredential(id=row.id, identity_id=row.identity_id, tenant_id=ctx.tenant_id)
