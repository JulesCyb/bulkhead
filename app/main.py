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

from app.api import agents, chat, health
from app.config import Settings, get_settings
from app.observability import setup_observability

log = logging.getLogger(__name__)

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


def check_auth_mode(settings: Settings) -> None:
    """Refuse to start outside dev/test with header auth — the guardrail lives in code,
    not only in the docs. With dev-headers, every request can impersonate any tenant."""
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
    setup_observability(settings)
    yield


async def handle_permission_error(request: Request, exc: Exception) -> JSONResponse:
    """ADR-0004: a failed `RequestContext.require_role` check answers 403, never 500 — clean and
    predictable regardless of which route, tool, or dependency called it. `PermissionError`
    carries no sensitive detail (only the missing role name), so it is safe to echo back."""
    message = str(exc) or "This action requires a role you don't have."
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={"error": "forbidden", "message": message},
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


def create_app() -> FastAPI:
    settings = get_settings()
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

    app.include_router(health.router)
    app.include_router(tenant_router)
    return app


app = create_app()
