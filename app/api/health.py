import logging

from fastapi import APIRouter, Depends, HTTPException

from app.db.guard import run_role_rls_guard

log = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


async def get_role_rls_guard():
    """The default guard dependency: the live `run_role_rls_guard` coroutine function itself,
    looked up fresh on every request (never memoized), so `/ready` re-runs the real query each
    time rather than reusing a cached result. A test substitutes this dependency
    (`app.dependency_overrides[get_role_rls_guard]`) with a fake guard while leaving
    `check_readiness`'s own generic-failure handling below untouched."""
    return run_role_rls_guard


async def check_readiness(guard=Depends(get_role_rls_guard)) -> None:  # noqa: B008
    """Runs the (possibly substituted) guard and turns any failure into a generic 503, logging
    the real reason server-side only — the response body must never name the offending role or
    table (ADR-0011: a probe response must not be usable to fingerprint the deployment)."""
    try:
        await guard()
    except Exception:
        log.exception("Readiness check failed: privileged role or missing forced RLS.")
        raise HTTPException(status_code=503, detail="not ready") from None


@router.get("/ready")
async def ready(_: None = Depends(check_readiness)) -> dict[str, str]:
    return {"status": "ready"}
