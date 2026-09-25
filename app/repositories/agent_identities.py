"""Agent-identity repository (ADR-0005, Spec 6 / #46): the only path to creating an agent
identity and its membership.

`app` has no `INSERT` grant on `control.identities` (ADR-0003 / #22) -- an agent identity is
created through `control.create_agent_identity`, a narrow `SECURITY DEFINER` function owned by
`app_owner` (migration 0032) that does exactly two things inside the caller's already-open
transaction: insert a `control.identities` row of kind `'agent'` (issuer/subject synthesized
inside the function, never taken from the caller, so nothing here can impersonate or collide with
a real person's identity), and insert its `agent`-role membership in the tenant the caller's own
session is already scoped to (`app.tenant_id`, set by `tenant_session(ctx)` before this ever
runs). Because the function reads that GUC rather than taking a tenant id argument, a call made
from tenant A's session has no parameter through which to name tenant B -- and the membership
INSERT still has to satisfy `memberships`' own `FORCE ROW LEVEL SECURITY` policy, the same trust
boundary every other tenant-scoped write in this codebase already relies on.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.context import RequestContext


class AgentIdentity(BaseModel):
    """What creating an agent identity returns: its own global identifier and the membership
    that admits it to the caller's tenant with role `agent`."""

    identity_id: UUID
    membership_id: UUID


class AgentIdentityRepository:
    async def create(
        self, session: AsyncSession, ctx: RequestContext, *, name: str
    ) -> AgentIdentity:
        """Create an agent identity plus its `agent`-role membership in `ctx.tenant_id`,
        atomically, via `control.create_agent_identity` (migration 0032). `ctx` is not otherwise
        consulted here -- the tenant comes from the already-set `app.tenant_id` GUC inside this
        same transaction, not from `ctx.tenant_id` passed as a value, matching how the function
        itself resolves it."""
        row = (
            (
                await session.execute(
                    text(
                        "SELECT identity_id, membership_id "
                        "FROM control.create_agent_identity(:name)"
                    ),
                    {"name": name},
                )
            )
            .mappings()
            .one()
        )
        return AgentIdentity(identity_id=row["identity_id"], membership_id=row["membership_id"])
