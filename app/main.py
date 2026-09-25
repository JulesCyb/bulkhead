"""FastAPI app: the agent backend as an API.

Start: uv run uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api import agents, chat, health
from app.config import Settings, get_settings
from app.observability import setup_observability

log = logging.getLogger(__name__)


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


def create_app() -> FastAPI:
    settings = get_settings()
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

    app.include_router(health.router)
    app.include_router(agents.router)
    app.include_router(chat.router)
    return app


app = create_app()
