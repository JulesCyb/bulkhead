"""FastAPI dependencies: context from the request, a tenant-bound DB session.

Every tenant-scoped route lives under `/v1/t/{tenant_id}/` (ADR-0012) -- the tenant id in the
URL path is the request's sole statement of intent; nothing else may name the tenant.

`get_context` is the HTTP adapter of `app/context_resolution.py` (#101), which owns the whole
chain under both `AUTH_MODE` values -- bearer parsing, token verification with per-issuer key and
algorithm pinning, audience against the path, identity, membership, suspension, and the means --
and returns either a `RequestContext` or a `ContextRejection`. This module only reads the
request's inputs, renders a rejection as an `HTTPException` (logging a forbidden one as the
`jwt_auth_forbidden` security event, never the token), and stashes the context on
`request.state` for the exception handlers and the request-id middleware in `app/main.py`.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import Depends, Header, HTTPException, Path, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app import context_resolution
from app.config import Settings, get_settings
from app.context import RequestContext
from app.context_resolution import FORBIDDEN_DETAIL, ContextRejection
from app.db.session import tenant_session
from app.jwt_verifier import KeySource
from app.token_verifier import AlgorithmSource

__all__ = [
    "FORBIDDEN_DETAIL",
    "Context",
    "Session",
    "get_algorithm_source",
    "get_context",
    "get_key_source",
    "get_session",
]

log = logging.getLogger(__name__)


def get_key_source(settings: Annotated[Settings, Depends(get_settings)]) -> KeySource:
    """Thin FastAPI alias of `app.context_resolution.key_source_for` (per-issuer key pinning).
    Override this dependency (`app.dependency_overrides[get_key_source] = ...`) to inject a
    JWKS-backed or test key source; never edit `app/jwt_verifier.py` to reach the network."""
    return context_resolution.key_source_for(settings)


def get_algorithm_source(settings: Annotated[Settings, Depends(get_settings)]) -> AlgorithmSource:
    """Thin FastAPI alias of `app.context_resolution.algorithm_source_for` (per-issuer algorithm
    pinning, the algorithm-confusion guard)."""
    return context_resolution.algorithm_source_for(settings)


def _render(rejection: ContextRejection, tenant_id: UUID) -> HTTPException:
    """The HTTP shape of a rejection: its status and client-visible detail; a forbidden one is
    also the one structured security-event line naming reason, tenant, issuer, and request id."""
    if rejection.is_security_event:
        log.warning(
            "request rejected: not authorized for this tenant",
            extra={
                "event": "jwt_auth_forbidden",
                "reason": rejection.reason.value,
                "tenant_id": str(tenant_id),
                "issuer": rejection.issuer or "",
                "request_id": rejection.request_id,
            },
        )
    return HTTPException(int(rejection.status), rejection.detail)


async def get_context(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    tenant_id: Annotated[UUID, Path()],
    key_source: Annotated[KeySource, Depends(get_key_source)],
    algorithm_source: Annotated[AlgorithmSource, Depends(get_algorithm_source)],
    x_identity_id: Annotated[str | None, Header()] = None,
    x_roles: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> RequestContext:
    outcome = await context_resolution.resolve_request_context(
        tenant_id=tenant_id,
        authorization=authorization,
        settings=settings,
        dev_identity_id=x_identity_id,
        dev_roles=x_roles,
        key_source=key_source,
        algorithm_source=algorithm_source,
    )
    if isinstance(outcome, ContextRejection):
        raise _render(outcome, tenant_id)
    # Stashed on request.state (not returned as a header here) so the ASGI middleware in
    # app/main.py can attach it to the response regardless of the route's return type, and so a
    # request that fails before a context exists never gets the header at all.
    request.state.request_id = outcome.request_id
    # Also stashed whole (S3-T1 / #26): `app.main.handle_permission_error` reads it back to log a
    # denied role check with identifiers only, without re-deriving them.
    request.state.context = outcome
    return outcome


Context = Annotated[RequestContext, Depends(get_context)]


async def get_session(ctx: Context) -> AsyncIterator[AsyncSession]:
    async with tenant_session(ctx) as session:
        yield session


Session = Annotated[AsyncSession, Depends(get_session)]
