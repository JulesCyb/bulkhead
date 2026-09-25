"""FastAPI dependencies: context from the request, a tenant-bound DB session.

AUTH_MODE=dev-headers reads X-Tenant-Id / X-Identity-Id / X-Roles from the headers — for local
development ONLY. Implement AUTH_MODE=jwt before production (verify the OIDC token, then
build the context from its claims). X-Roles is a development-only convenience that lets a
caller assert its own roles directly; it has no production equivalent — under AUTH_MODE=jwt,
roles come from exactly one place, the caller's membership row, never from a client-supplied
header.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.context import RequestContext
from app.db.session import tenant_session


async def get_context(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    x_tenant_id: Annotated[str | None, Header()] = None,
    x_identity_id: Annotated[str | None, Header()] = None,
    x_roles: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> RequestContext:
    if settings.auth_mode == "dev-headers":
        if not x_tenant_id or not x_identity_id:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "X-Tenant-Id and X-Identity-Id are missing (AUTH_MODE=dev-headers)",
            )
        try:
            tenant_id, identity_id = UUID(x_tenant_id), UUID(x_identity_id)
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid UUID in header") from exc
        roles = frozenset(r.strip() for r in (x_roles or "").split(",") if r.strip())
        ctx = RequestContext(tenant_id=tenant_id, identity_id=identity_id, roles=roles)
        # Stashed on request.state (not returned as a header here) so the ASGI middleware in
        # app/main.py can attach it to the response regardless of the route's return type
        # (JSONResponse, StreamingResponse, or the chat endpoint's Vercel AI stream), and so a
        # request that fails before a context exists never gets the header at all.
        request.state.request_id = ctx.request_id
        return ctx

    # AUTH_MODE=jwt: verify the bearer token (signature, issuer, expiry) and read the claims.
    # Deliberately not implemented "somehow" — wrong auth is worse than none.
    raise HTTPException(
        status.HTTP_501_NOT_IMPLEMENTED,
        "AUTH_MODE=jwt is not implemented yet (see app/deps.py)",
    )


Context = Annotated[RequestContext, Depends(get_context)]


async def get_session(ctx: Context) -> AsyncIterator[AsyncSession]:
    async with tenant_session(ctx) as session:
        yield session


Session = Annotated[AsyncSession, Depends(get_session)]
