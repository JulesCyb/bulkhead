"""Admin-only routes for agent identities and their credentials (ADR-0005, Spec 6 / #46).

Mirrors `app/api/memberships.py`'s pattern exactly: each route calls a tool function from
`app/tools/agent_identities.py` that checks the caller's role first, so the enforcement lives in
exactly one place. A non-admin membership never reaches a repository -- it is refused with a 403
from the registered exception handler (`app.main.handle_permission_error`), never a crash.

Credential routes are addressed by credential id directly (`/agent-credentials/...`), not nested
under an identity, to keep the static `GET /agent-credentials` listing route from ever competing
with a dynamic `/agent-identities/{identity_id}/...` segment.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter
from pydantic import BaseModel

from app.deps import Context
from app.repositories.agent_credentials import CredentialRecord, IssuedCredential
from app.repositories.agent_identities import AgentIdentity
from app.tools.agent_identities import (
    create_agent_identity,
    issue_agent_credential,
    list_agent_credentials,
    revoke_agent_credential,
)

router = APIRouter(tags=["agent-identities"])


class CreateAgentIdentityRequest(BaseModel):
    name: str


class IssueAgentCredentialRequest(BaseModel):
    name: str


class CredentialListResponse(BaseModel):
    credentials: list[CredentialRecord]


class RevokeCredentialResponse(BaseModel):
    revoked: bool


@router.post("/agent-identities", response_model=AgentIdentity)
async def create_agent_identity_route(
    body: CreateAgentIdentityRequest, ctx: Context
) -> AgentIdentity:
    return await create_agent_identity(ctx, name=body.name)


@router.post("/agent-identities/{identity_id}/credentials", response_model=IssuedCredential)
async def issue_agent_credential_route(
    identity_id: UUID, body: IssueAgentCredentialRequest, ctx: Context
) -> IssuedCredential:
    return await issue_agent_credential(ctx, identity_id=identity_id, name=body.name)


@router.get("/agent-credentials", response_model=CredentialListResponse)
async def list_agent_credentials_route(ctx: Context) -> CredentialListResponse:
    records = await list_agent_credentials(ctx)
    return CredentialListResponse(credentials=records)


@router.post("/agent-credentials/{credential_id}/revoke", response_model=RevokeCredentialResponse)
async def revoke_agent_credential_route(
    credential_id: UUID, ctx: Context
) -> RevokeCredentialResponse:
    revoked = await revoke_agent_credential(ctx, credential_id=credential_id)
    return RevokeCredentialResponse(revoked=revoked)
