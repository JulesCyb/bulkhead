"""Agent-credential token exchange route (ADR-0005, ADR-0012, Spec 6 / #47).

Deliberately outside `app.deps.Context`/the bearer-token dependency: this is how an agent
identity's automation obtains a token in the first place, so there is no token yet to check. The
tenant still comes only from the URL path (ADR-0012); the credential's own secret is the one
thing that authenticates the caller.

An unknown credential, a wrong secret, and a revoked credential all produce the exact same
response (status and body) -- `app/agent_credential_exchange.py` collapses all three to one
exception, and this route maps it to one generic detail, never revealing which part was wrong.
"""

from __future__ import annotations

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel

from app.agent_credential_exchange import AgentCredentialExchangeError, exchange_agent_credential
from app.config import Settings, get_settings

log = logging.getLogger(__name__)

router = APIRouter(prefix="/agent-tokens", tags=["agent-tokens"])

# Every failure kind (unknown identifier, wrong secret, revoked credential) returns exactly this
# body -- never distinguishable from one another, matching how AUTH_MODE=jwt's own rejections
# stay generic (app/deps.py::FORBIDDEN_DETAIL).
INVALID_CREDENTIAL_DETAIL = "Invalid credential."


class AgentTokenRequest(BaseModel):
    public_id: str
    secret: str


class AgentTokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int


@router.post("", response_model=AgentTokenResponse)
async def exchange_credential_for_token(
    tenant_id: Annotated[UUID, Path()],
    body: AgentTokenRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> AgentTokenResponse:
    try:
        issued = await exchange_agent_credential(
            tenant_id=tenant_id, public_id=body.public_id, secret=body.secret, settings=settings
        )
    except AgentCredentialExchangeError as exc:
        log.warning(
            "Agent credential exchange refused",
            extra={"event": "agent_credential_exchange_refused", "tenant_id": str(tenant_id)},
        )
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, INVALID_CREDENTIAL_DETAIL) from exc
    return AgentTokenResponse(
        access_token=issued.access_token,
        token_type=issued.token_type,
        expires_in=issued.expires_in,
    )
