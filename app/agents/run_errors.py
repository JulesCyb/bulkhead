"""The one error table for a run (spec A3 / #93, #107): which failures of a prepared run
(`app.agents.run`) every entry point maps to a documented, client-visible error, and how each
transport renders it -- an HTTP status with a JSON `detail`, or a terminal SSE `event: error`.

`app/api/agents.py` uses both renderings; `app/api/chat.py` uses the HTTP one for a failure before
its stream starts (once streaming, the chat run ends with a Vercel AI SDK `error` chunk of its
own, `app.agents.run`). A failure not in the table (a
`NotFoundInTenant`, a `RoleRequired`, anything unexpected) is not mapped here: it propagates to
the handlers `app.main` registers, exactly as before.

| Failure | Status | `error` |
|---|---|---|
| `TenantSuspendedError` (ADR-0010) | 403 | `forbidden` (the generic forbidden detail) |
| `UsageLimitExceeded` (run limit, ADR-0009) | 429 | `run_limit_exceeded` |
| `RunDeadlineExceeded` (run limit, ADR-0009) | 504 | `run_deadline_exceeded` |
| `ROUTING_ERRORS` (ADR-0008) | 503 | `content_routing_unavailable` |
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from pydantic_ai.exceptions import UsageLimitExceeded

from app.context_resolution import FORBIDDEN_DETAIL
from app.db.session import TenantSuspendedError
from app.gateway_credentials import GatewayCredentialUnavailable
from app.llm import ModelNotAllowedForResidency
from app.residency import ResidencyUnresolved
from app.run_limits import RunDeadlineExceeded

# Raised by preparation (`app.agents.run.prepare_run`) when the requesting tenant's residency
# cannot be routed at all -- unresolved/unknown residency, a chosen model outside its allow-list,
# or a missing gateway credential (Spec 8 / #61, ADR-0008). Every one of these is a fail-closed
# routing failure, never a fallback to a default route, so all three map to the same clear,
# distinct error rather than a raw exception or a silent default.
ROUTING_ERRORS = (ResidencyUnresolved, ModelNotAllowedForResidency, GatewayCredentialUnavailable)

# Every failure the table below maps -- `except MAPPED_RUN_ERRORS` at a route, then render.
MAPPED_RUN_ERRORS: tuple[type[BaseException], ...] = (
    TenantSuspendedError,
    UsageLimitExceeded,
    RunDeadlineExceeded,
    *ROUTING_ERRORS,
)


@dataclass(frozen=True, slots=True)
class RunError:
    """One mapped failure: its HTTP status, its machine-readable `error`, a message, and the
    HTTP `detail` body (a bare string for the generic 403, `{"error", "message"}` otherwise)."""

    status_code: int
    error: str
    message: str
    detail: Any


def map_run_error(exc: BaseException) -> RunError:
    """The table row for `exc`, which must be one of `MAPPED_RUN_ERRORS`."""
    if isinstance(exc, TenantSuspendedError):
        return RunError(403, "forbidden", FORBIDDEN_DETAIL, FORBIDDEN_DETAIL)
    if isinstance(exc, UsageLimitExceeded):
        status_code, error = 429, "run_limit_exceeded"
    elif isinstance(exc, RunDeadlineExceeded):
        status_code, error = 504, "run_deadline_exceeded"
    elif isinstance(exc, ROUTING_ERRORS):
        status_code, error = 503, "content_routing_unavailable"
    else:
        raise TypeError(f"{type(exc).__name__} is not a mapped run error") from exc
    message = str(exc)
    return RunError(status_code, error, message, {"error": error, "message": message})


def run_error_http_exception(exc: BaseException) -> HTTPException:
    """`exc` (one of `MAPPED_RUN_ERRORS`) as the HTTP error a JSON endpoint raises."""
    mapped = map_run_error(exc)
    return HTTPException(status_code=mapped.status_code, detail=mapped.detail)


def sse(data: str) -> str:
    """Frame text as one SSE event. Multi-line deltas become multiple data: lines,
    which the client reassembles with newlines — raw interpolation would silently
    drop every line lacking the data: prefix.
    """
    return "".join(f"data: {line}\n" for line in data.split("\n")) + "\n"


def sse_error(error: str, message: str) -> str:
    """Frame a terminal `event: error` SSE event — the mapped, clean end to a run, distinct from
    a silent stop or a raw exception. `message` is deliberately generic for a deadline (see
    `RunDeadlineExceeded`); it never carries the underlying provider's own error text.
    """
    payload = json.dumps({"error": error, "message": message})
    return f"event: error\ndata: {payload}\n\n"


def run_error_sse_event(exc: BaseException) -> str:
    """`exc` (one of `MAPPED_RUN_ERRORS`) as the terminal SSE error event a stream ends with --
    by the time a stream fails, its response has already started (status 200, headers sent), so
    the failure must become an event, never a raw exception on an already-started stream."""
    mapped = map_run_error(exc)
    return sse_error(mapped.error, mapped.message)


__all__ = [
    "MAPPED_RUN_ERRORS",
    "ROUTING_ERRORS",
    "RunError",
    "map_run_error",
    "run_error_http_exception",
    "run_error_sse_event",
    "sse",
    "sse_error",
]
