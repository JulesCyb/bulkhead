"""FastAPI app: the agent backend as an API.

Start: uv run uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api import agents, chat, health
from app.config import Settings, get_settings
from app.db.guard import run_role_rls_guard
from app.observability import setup_observability

log = logging.getLogger(__name__)


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
    setup_observability(settings)
    # Fail-closed startup guard (issue #15 / ADR-0011): refuses to ever accept traffic while
    # connected as a superuser/BYPASSRLS role, or while any public-schema table lacks forced
    # RLS. Looked up by name (not bound at import time) so a test can monkeypatch
    # `app.main.run_role_rls_guard` and drive this lifespan directly to prove the guard runs
    # here independent of the readiness endpoint's own dependency.
    await run_role_rls_guard()
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    # Runs at construction time, not only inside the lifespan above (issue #15 / ADR-0011): no
    # way of building the application object — including a test harness that never fires ASGI
    # lifespan events — can skip this guard.
    check_auth_mode(settings)
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    origins = settings.cors_origin_list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        # Credentialed requests must never combine with a wildcard origin.
        allow_credentials="*" not in origins,
        allow_methods=["*"],
        allow_headers=["*"],
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
