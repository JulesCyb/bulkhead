"""Tool functions for agent identities and their credentials (ADR-0005, Spec 6 / #46) -- a
shared module for the admin HTTP routes and, later, any MCP exposure of the same actions.

Each function checks `ctx.require_role("admin")` first, before any data access -- exactly the
pattern `app/tools/memberships.py` established (ADR-0004, S3-T1 / #26): a failed check raises
`PermissionError`, turned into a 403 by the registered exception handler
(`app.main.handle_permission_error`), never a 500.

`issue_agent_credential` additionally relies on `AgentCredentialRepository.create` to raise
`app.repositories.agent_credentials.UnknownAgentIdentity` (never letting a bad `identity_id`
reach the database as anything other than that one exception, see that module's docstring) --
turned into a 404 by `app.main.handle_unknown_agent_identity`, identical regardless of whether
`identity_id` names nothing at all, another tenant's identity, or a person's identity in this
tenant.
"""

from __future__ import annotations

from uuid import UUID

from app.context import RequestContext
from app.db.session import tenant_session
from app.repositories.agent_credentials import (
    AgentCredentialRepository,
    CredentialRecord,
    IssuedCredential,
)
from app.repositories.agent_identities import AgentIdentity, AgentIdentityRepository


async def create_agent_identity(ctx: RequestContext, *, name: str) -> AgentIdentity:
    """Create an agent identity for the caller's tenant, with a membership of role `agent`.
    Admin-only."""
    ctx.require_role("admin")
    async with tenant_session(ctx) as session:
        return await AgentIdentityRepository().create(session, ctx, name=name)


async def issue_agent_credential(
    ctx: RequestContext, *, identity_id: UUID, name: str
) -> IssuedCredential:
    """Issue a named credential for an agent identity in the caller's tenant. The plaintext
    secret is returned exactly once, here -- never again, by any later call. Admin-only."""
    ctx.require_role("admin")
    async with tenant_session(ctx) as session:
        return await AgentCredentialRepository().create(
            session, ctx, identity_id=identity_id, name=name
        )


async def list_agent_credentials(ctx: RequestContext) -> list[CredentialRecord]:
    """List every credential issued in the caller's tenant -- metadata only, never a secret or
    its hash. Admin-only."""
    ctx.require_role("admin")
    async with tenant_session(ctx) as session:
        return await AgentCredentialRepository().list_for_tenant(session, ctx)


async def revoke_agent_credential(ctx: RequestContext, *, credential_id: UUID) -> bool:
    """Revoke a credential in the caller's tenant immediately. Idempotent: revoking an
    already-revoked (or unknown, or cross-tenant) credential returns False. Admin-only."""
    ctx.require_role("admin")
    async with tenant_session(ctx) as session:
        return await AgentCredentialRepository().revoke(session, ctx, credential_id=credential_id)
