"""Per-membership request limit on the agent-facing routes.

A single-process, best-effort backstop keyed on the `(tenant_id, user_id)` pair already
carried by every `RequestContext` — it exists to stop a stuck client or a scripted retry
loop from one member before it can dent the tenant's whole budget at the model gateway, not
to be a precise or distributed rate limiter.

**This is not the enforcement of record.** The model gateway's own per-tenant budget is —
this limiter only buys time before that budget is touched at all, and it does so only within
this one process: a deployment running more than one replica needs a shared store (Redis or
similar) for the same guarantee across replicas, which is a known, named limitation and not
solved here.

Exceeding this limit is distinct from, and reported separately from:
- a run-limit error (too many model requests/tool calls within a single agent run), and
- a gateway-budget error (the tenant's spend ceiling being exhausted at the model gateway),
so a caller can tell which of the three protections it tripped.
"""

from __future__ import annotations

import time
from collections import deque
from functools import lru_cache
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, status

from app.config import Settings, get_settings
from app.deps import Context


class RequestLimitExceeded(HTTPException):
    """Raised when a `(tenant_id, user_id)` pair exceeds its request-limit window.

    A 429 with a clearly-labeled body (`error: "request_limit_exceeded"`) so a caller can
    distinguish this from a run-limit error or a gateway-budget error, both reported
    differently at their own seams.
    """

    def __init__(self, retry_after_seconds: float) -> None:
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "request_limit_exceeded",
                "message": (
                    "Too many requests from this tenant/user in the current window. "
                    "This is a per-process backstop, not the tenant's budget at the model "
                    "gateway — slow down and retry after the window given."
                ),
                "retry_after_seconds": round(max(retry_after_seconds, 0.0), 3),
            },
        )


class RequestLimiter:
    """In-process fixed/sliding-window limiter, keyed by an arbitrary hashable key.

    Deliberately simple: a `deque` of monotonic timestamps per key, trimmed to the
    configured window on every check. Safe under FastAPI's single-threaded event loop
    without extra locking because `check()` never awaits.
    """

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._windows: dict[tuple[UUID, UUID], deque[float]] = {}

    def check(self, key: tuple[UUID, UUID], now: float | None = None) -> None:
        """Record one request for `key`, raising `RequestLimitExceeded` if it is over limit."""
        now = time.monotonic() if now is None else now
        window = self._windows.setdefault(key, deque())
        cutoff = now - self._window_seconds
        while window and window[0] <= cutoff:
            window.popleft()
        if len(window) >= self._max_requests:
            retry_after = window[0] + self._window_seconds - now
            raise RequestLimitExceeded(retry_after_seconds=retry_after)
        window.append(now)


@lru_cache
def _limiter_for(max_requests: int, window_seconds: float) -> RequestLimiter:
    """One limiter per distinct (max, window) configuration — effectively a singleton in
    production, where Settings is fixed for the process's lifetime."""
    return RequestLimiter(max_requests=max_requests, window_seconds=window_seconds)


async def enforce_request_limit(
    ctx: Context,
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """FastAPI dependency: raises `RequestLimitExceeded` (429) once the calling
    `(tenant_id, user_id)` pair exceeds `Settings.request_limit_max` within
    `Settings.request_limit_window_seconds`. A different pair — another member of the same
    tenant, or a member of another tenant — is tracked in its own window and unaffected.
    """
    limiter = _limiter_for(settings.request_limit_max, settings.request_limit_window_seconds)
    limiter.check((ctx.tenant_id, ctx.user_id))


RequestLimit = Annotated[None, Depends(enforce_request_limit)]
