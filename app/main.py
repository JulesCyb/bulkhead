"""FastAPI app: the agent backend as an API.

Start: uv run uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import agent_identities, agent_tokens, agents, chat, health, memberships
from app.config import Settings, get_settings
from app.context import RoleRequired
from app.db.guard import run_role_rls_guard
from app.mcp.server import build_streamable_http_app, check_mcp_mode
from app.observability import setup_observability
from app.repositories.errors import NotFoundInTenant
from app.startup_checks import run_startup_checks

log = logging.getLogger(__name__)

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


def check_auth_mode(settings: Settings) -> None:
    """Refuse to start outside dev/test with header auth — the guardrail lives in code,
    not only in the docs. With dev-headers, every request can impersonate any tenant.

    Called both at application-object construction time (`create_app`, below) and again inside
    the ASGI lifespan (issue #15 / ADR-0011): a way of launching the process that never fires
    lifespan events (e.g. some test harnesses) still hits this guard at construction, and a way
    that constructs the app once and only later changes what `get_settings()` returns still hits
    it again at lifespan startup — no path can skip it."""
    if settings.auth_mode != "dev-headers":
        return
    if settings.environment not in ("dev", "test"):
        raise RuntimeError(
            "AUTH_MODE=dev-headers is for local development only. "
            "Implement AUTH_MODE=jwt (app/deps.py) or set ENVIRONMENT=dev."
        )
    log.warning("AUTH_MODE=dev-headers active — local development only.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    check_auth_mode(settings)
    # The MCP transport's own fail-closed startup guard (issue #48 / ADR-0005) runs alongside
    # check_auth_mode above -- same reasoning, same moment: a way of launching the process that
    # never fires lifespan events must not be the only thing standing between a half-finished
    # MCP_TRANSPORT configuration and a request.
    check_mcp_mode(settings)
    # Fail-closed residency/model-allow-list guard (issue #59 / ADR-0008), same reasoning as
    # check_auth_mode above: run again here, independent of the construction-time call in
    # create_app, so a way of launching the process that constructs the app once and only later
    # changes what get_settings() returns still hits it at lifespan startup.
    run_startup_checks(settings)
    setup_observability(settings)
    # Fail-closed startup guard (issue #15 / ADR-0011): refuses to ever accept traffic while
    # connected as a superuser/BYPASSRLS role, or while any public-schema table lacks forced
    # RLS. Looked up by name (not bound at import time) so a test can monkeypatch
    # `app.main.run_role_rls_guard` and drive this lifespan directly to prove the guard runs
    # here independent of the readiness endpoint's own dependency.
    await run_role_rls_guard()
    yield


async def handle_permission_error(request: Request, exc: Exception) -> JSONResponse:
    """ADR-0004: a failed `RequestContext.require_role` check answers 403, never 500 — clean and
    predictable regardless of which route, tool, or dependency called it. `PermissionError`
    carries no sensitive detail (only the missing role name), so it is safe to echo back.

    S3-T1 / #26: also logs the denial with identifiers only (tenant, identity, the required
    role) — never request content — so a pattern of repeated denials is visible in the
    application logs. `request.state.context` is the `RequestContext` `app.deps.get_context`
    already resolved for this request, in both dev-headers and jwt mode.

    The required role comes from `exc.required_role` when `exc` is a `RoleRequired`
    (`app/context.py`) — never re-derived from `str(exc)` with a regex, which broke the moment
    the message text changed. A `PermissionError` raised from somewhere else (e.g.
    `ConversationOwnershipError`) carries no such attribute, so `required_role` is `None` for it,
    exactly as before.
    """
    message = str(exc) or "This action requires a role you don't have."
    ctx = getattr(request.state, "context", None)
    required_role = exc.required_role if isinstance(exc, RoleRequired) else None
    log.warning(
        "Role check denied",
        extra={
            "event": "role_check_denied",
            "tenant_id": str(ctx.tenant_id) if ctx is not None else None,
            "identity_id": str(ctx.identity_id) if ctx is not None else None,
            "required_role": required_role,
        },
    )
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={"error": "forbidden", "message": message},
    )


async def handle_not_found_in_tenant(request: Request, exc: Exception) -> JSONResponse:
    """The one shared answer for every repository condition meaning "nothing in the caller's own
    tenant matches" (`app.repositories.errors.NotFoundInTenant` and its subclasses) -- fail
    closed, never leak existence across tenants.

    Finding from the 2026-09-25 review of #46: `issue_agent_credential` used to check only that
    the *caller* was an admin, never that `identity_id` actually named an agent identity in the
    caller's own tenant -- so an admin could mint a (unusable, but real) credential row for
    another tenant's identity, or a person's, and the response would tell them which. That became
    `UnknownAgentIdentity` (`app/repositories/agent_credentials.py`). The same review, sweeping
    for the same shape elsewhere, found `NotAnAgentMembership`
    (`app/repositories/standing_grants.py`) had the identical reasoning in its own docstring but
    no handler at all -- it fell through to the generic 500, which is itself a leak (500 vs 404
    already tells a caller "this id exists somewhere" a 404 wouldn't). Both are now
    `NotFoundInTenant` subclasses, caught by this one handler registered on the base class
    (Starlette dispatches by walking the exception's MRO, so a new subclass needs no new handler
    registered here) -- this is the pattern a new repository should reuse for its own "not in
    this tenant" condition, never a bespoke per-exception handler.

    This gives the same answer -- a 404 with a fixed body -- for every reason the raised
    exception's type can occur: an id that names nothing, a cross-tenant id, or one that exists
    here but fails some other in-tenant invariant (e.g. a membership with the wrong role).
    `str(exc)` (which may name the real id) is logged for operators; the client only ever sees
    `exc.public_message`, the fixed sentence the raising exception class itself owns -- so it
    never learns anything an unknown id wouldn't also tell it."""
    log.info("%s refused: %s", type(exc).__name__, exc)
    message = getattr(exc, "public_message", NotFoundInTenant.public_message)
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "not_found", "message": message},
    )


async def handle_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    """S1-T7 / #17: every other unhandled exception. The client gets back only an id it can hand
    to support — never a SQL statement, a provider's raw error body, or a stack trace. The real
    failure text goes to the server's own log, tagged with that same id.

    `request.state.request_id` is set by `app.deps.get_context` as soon as a context exists; a
    failure before that (missing/invalid identity header) never reaches here as an unhandled
    exception in the first place, but a fresh id is minted defensively rather than leaving the
    field empty.
    """
    request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
    log.error("Unhandled exception for request_id=%s", request_id, exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_error",
            "message": "Something went wrong. Include this id if you contact support.",
            "request_id": request_id,
        },
    )


class ChunkedBodySizeLimitMiddleware:
    """Enforces a request body size cap while the body streams in, not only against a declared
    `Content-Length` header.

    A client using chunked transfer encoding sends no `Content-Length` at all, so a check against
    that header (see app/api/chat.py's own cap) never fires no matter how large the body actually
    is. This is a pure-ASGI middleware (not `@app.middleware("http")`/`BaseHTTPMiddleware`, which
    buffers the whole body before your code ever sees it) so it can reject the request as soon as
    the running total crosses the cap, before the rest of the body is ever read off the wire.

    Only applied to the path this ticket calls out (POST .../api/chat); every other route is
    passed straight through untouched.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int, path_suffix: str) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.path_suffix = path_suffix

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or not scope["path"].endswith(self.path_suffix)
        ):
            await self.app(scope, receive, send)
            return

        buffered: list[Message] = []
        total = 0
        while True:
            message = await receive()
            buffered.append(message)
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_bytes:
                    response = JSONResponse(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        content={
                            "error": "request_too_large",
                            "message": "Request body too large",
                        },
                    )
                    await response(scope, receive, send)
                    return
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                break

        index = 0

        async def replay_receive() -> Message:
            nonlocal index
            if index < len(buffered):
                message = buffered[index]
                index += 1
                return message
            return await receive()

        await self.app(scope, replay_receive, send)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    # Runs at construction time, not only inside the lifespan above (issue #15 / ADR-0011): no
    # way of building the application object — including a test harness that never fires ASGI
    # lifespan events — can skip this guard.
    check_auth_mode(settings)
    # Same reasoning, for the MCP transport's own guard (issue #48 / ADR-0005): construction time,
    # not only lifespan -- see the matching call in `lifespan` above.
    check_mcp_mode(settings)
    # Same reasoning, for the residency/model-allow-list guard (issue #59 / ADR-0008): a
    # deliberately mismatched configuration must never construct an application object that
    # could later accept a request, regardless of whether lifespan ever fires.
    run_startup_checks(settings)
    # S1-T7 / #17: interactive API documentation is reachable only in development and test —
    # not a production-like deployment (docs/reviews/2026-09-12-security-review.md). Same
    # dev/test allow-list as check_auth_mode above.
    docs_enabled = settings.environment in ("dev", "test")
    app = FastAPI(
        title=settings.app_name,
        lifespan=lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )
    app.add_exception_handler(PermissionError, handle_permission_error)
    app.add_exception_handler(NotFoundInTenant, handle_not_found_in_tenant)
    app.add_exception_handler(Exception, handle_unhandled_exception)
    origins = settings.cors_origin_list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        # Credentialed requests must never combine with a wildcard origin.
        allow_credentials="*" not in origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(
        ChunkedBodySizeLimitMiddleware, max_bytes=chat.MAX_BODY_BYTES, path_suffix="/api/chat"
    )

    @app.middleware("http")
    async def add_request_id_header(request: Request, call_next):
        """Every authenticated response carries the context's own request id back as a header
        (app/deps.py's get_context stashes it on request.state), so a member's bug report or
        an on-call engineer's log line can be tied to the exact request without reading any
        conversation content. A request that fails before a context exists (e.g. missing
        identity/tenant headers) never reaches this far with request.state.request_id set, so
        its existing error response is unchanged."""
        response = await call_next(request)
        request_id = getattr(request.state, "request_id", None)
        if request_id:
            response.headers["X-Request-Id"] = request_id
        return response

    # ADR-0012: every tenant-scoped route lives under /v1/t/{tenant_id}/ — the path segment is
    # the request's sole statement of intent. `health` is not tenant-scoped and stays outside it.
    tenant_router = APIRouter(prefix="/v1/t/{tenant_id}")
    tenant_router.include_router(agents.router)
    tenant_router.include_router(chat.router)
    tenant_router.include_router(memberships.router)
    tenant_router.include_router(agent_identities.router)
    tenant_router.include_router(agent_tokens.router)

    app.include_router(health.router)
    app.include_router(tenant_router)

    # The MCP server's networked transport (issue #49 / ADR-0005): mounted inside this same API
    # service, under the tenant's own path prefix (ADR-0012) -- never a second service, and never
    # reachable at all under the stdio (local-development) transport, which never touches this
    # FastAPI app in the first place. `check_mcp_mode` above has already refused to let this
    # branch be reached with `mcp_transport == "streamable-http"` and no verifier configured.
    if settings.mcp_transport == "streamable-http":
        app.mount(
            "/v1/t/{tenant_id}/mcp", build_streamable_http_app(settings), name="mcp-streamable-http"
        )

    return app


app = create_app()
