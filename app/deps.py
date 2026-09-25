"""FastAPI dependencies: context from the request, a tenant-bound DB session.

Every tenant-scoped route lives under `/v1/t/{tenant_id}/` (ADR-0012) — the tenant id in the
URL path is the request's sole statement of intent; nothing else may name the tenant, so no
stray client-supplied header can override it.

AUTH_MODE=dev-headers reads X-Identity-Id / X-Roles from the headers — for local development
ONLY. X-Roles is a development-only convenience that lets a caller assert its own roles
directly; it has no production equivalent — under AUTH_MODE=jwt, roles come from exactly one
place, the caller's membership row, never from a client-supplied header.

AUTH_MODE=jwt (issue #24, ADR-0003, ADR-0012) checks, in this order:

1. A bearer token is present and its signature, issuer, and expiry verify (`app/jwt_verifier.py`,
   no database involved) — otherwise 401 Unauthorized. An unconfigured token issuer or
   verification key also lands here: nothing to verify a signature against is the same as no
   valid signature.
2. The token's audience names the same tenant as the URL path — otherwise 403 Forbidden.
3. The (issuer, subject) the token names is on file as a known identity
   (`control.identity_lookup`) — otherwise 403 Forbidden.
4. That identity has a membership in that tenant (`memberships`, read inside the tenant's own
   context — no RLS bypass needed: no row means 403) — otherwise 403 Forbidden. This is also
   what makes a path naming a tenant that does not exist at all behave identically to one naming
   a tenant the caller simply isn't a member of: neither has a membership row.

Checks 1-4 above are delegated to `app/token_verifier.py::verify_tenant_token` (issue #44) — a
module with no FastAPI/HTTP dependency of its own, so a second caller (the MCP transport, Spec 6)
can reuse the exact same check instead of a second implementation of it.

5. The tenant is not suspended (`control.tenant_auth_settings`) — otherwise 403 Forbidden. This
   check is *not* part of the shared module above; it is enforced here, directly, the same way
   every other place that resolves a context enforces it (issue #69).

Every step-2-through-5 rejection returns the exact same generic body (`FORBIDDEN_DETAIL` below)
and is logged as a security event (`app/deps.py`'s `log`) naming the reason, the tenant id, the
issuer, and a request id — never the raw token — so the specific reason is discoverable only
from the log, never from the response.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import Depends, Header, HTTPException, Path, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.context import RequestContext
from app.db.session import control_session, tenant_session
from app.jwt_verifier import KeySource, TokenVerificationError
from app.repositories.control import TenantAuthSettingsRepository
from app.token_verifier import (
    TenantTokenVerificationError,
    VerificationFailureReason,
    verify_tenant_token,
)

log = logging.getLogger(__name__)

# Every forbidden (403) branch under AUTH_MODE=jwt returns exactly this body, regardless of which
# check failed — the reason is discoverable only from the structured log line, never the
# response (see module docstring and the ASGI-seam tests in tests/test_jwt_auth.py).
FORBIDDEN_DETAIL = "Not authorized for this tenant."


def get_key_source(settings: Annotated[Settings, Depends(get_settings)]) -> KeySource:
    """The default key source: a single process-wide verification key (interim, ADR-0003's
    "one operator-run identity provider" case) — never a network JWKS fetch. A customer-owned
    identity provider needs a real JWKS-backed KeySource; override this dependency
    (`app.dependency_overrides[get_key_source] = ...`, exactly how tests inject their own),
    never edit `app/jwt_verifier.py` to make it reach the network itself.
    """

    def _source(issuer: str, kid: str | None) -> str:
        if settings.jwt_verification_key is None:
            raise TokenVerificationError("no verification key configured")
        return settings.jwt_verification_key.get_secret_value()

    return _source


def _log_forbidden(*, reason: str, tenant_id: UUID, issuer: str, request_id: str) -> HTTPException:
    """Logs the one structured security-event line every forbidden branch produces, then returns
    (does not raise) the generic HTTPException the caller raises itself — keeping `raise
    _log_forbidden(...)` readable at each call site."""
    log.warning(
        "AUTH_MODE=jwt request rejected",
        extra={
            "event": "jwt_auth_forbidden",
            "reason": reason,
            "tenant_id": str(tenant_id),
            "issuer": issuer,
            "request_id": request_id,
        },
    )
    return HTTPException(status.HTTP_403_FORBIDDEN, FORBIDDEN_DETAIL)


async def _get_jwt_context(
    request: Request,
    settings: Settings,
    tenant_id: UUID,
    key_source: KeySource,
    authorization: str | None,
) -> RequestContext:
    request_id = uuid.uuid4().hex

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or malformed bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or malformed bearer token")

    try:
        resolved = await verify_tenant_token(
            token,
            tenant_id=tenant_id,
            key_source=key_source,
            default_issuer=settings.default_identity_issuer,
            algorithms=(settings.jwt_algorithm,),
        )
    except TenantTokenVerificationError as exc:
        if exc.reason is VerificationFailureReason.INVALID_OR_EXPIRED:
            detail = (
                "No token issuer configured" if exc.issuer is None else "Invalid or expired token"
            )
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail) from None
        raise _log_forbidden(
            reason=exc.reason.value,
            tenant_id=tenant_id,
            issuer=exc.issuer or "",
            request_id=request_id,
        ) from None

    # Tenant suspension is *not* part of the shared verifier above -- it is enforced here,
    # directly, the same way every other place that resolves a context enforces it (issue #69).
    # A tenant with no control-plane row at all (never marked dedicated/suspended, or not
    # created in the control plane yet) is treated as not suspended (ADR-0002, app/db/session.py).
    async with control_session() as session:
        auth_settings = await TenantAuthSettingsRepository().get(
            session, tenant_id=tenant_id, default_issuer=settings.default_identity_issuer
        )
    if auth_settings is not None and auth_settings.suspended:
        raise _log_forbidden(
            reason="tenant_suspended",
            tenant_id=tenant_id,
            issuer=resolved.issuer,
            request_id=request_id,
        )

    ctx = RequestContext(
        tenant_id=tenant_id,
        identity_id=resolved.identity_id,
        roles=frozenset({resolved.role}),
        request_id=request_id,
    )
    request.state.request_id = ctx.request_id
    return ctx


async def get_context(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    tenant_id: Annotated[UUID, Path()],
    key_source: Annotated[KeySource, Depends(get_key_source)],
    x_identity_id: Annotated[str | None, Header()] = None,
    x_roles: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> RequestContext:
    if settings.auth_mode == "dev-headers":
        if not x_identity_id:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "X-Identity-Id is missing (AUTH_MODE=dev-headers)",
            )
        try:
            identity_id = UUID(x_identity_id)
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

    return await _get_jwt_context(request, settings, tenant_id, key_source, authorization)


Context = Annotated[RequestContext, Depends(get_context)]


async def get_session(ctx: Context) -> AsyncIterator[AsyncSession]:
    async with tenant_session(ctx) as session:
        yield session


Session = Annotated[AsyncSession, Depends(get_session)]
