"""Process-wide answer to "which database engine serves this alias" (ADR-0002, Spec 10 / #74).

`app.db.session` owns the one engine every pooled tenant already shares
(`Settings.database_url`); this module owns the registry that sits above it. The pooled default
alias (`POOLED_ALIAS`) is always available and resolves to that same shared engine -- it is
never looked up through a secret file. Any other alias names a tenant's dedicated database: the
first request for it builds an `AsyncEngine` from a connection string read out of a tenant-secret
file keyed by that alias (ADR-0011's file-per-alias mechanism -- never a value on `Settings` or
on `tenants.settings`), and the engine is cached here for the life of the process. A lock per
alias makes sure concurrent first requests for the same brand-new alias build exactly one engine,
never a leaked duplicate connection pool. An alias with no matching secret file raises
`UnknownDatabaseAliasError` -- it never silently falls back to the pooled engine.

This module does not itself decide which alias a tenant uses (that is the control-plane lookup
`tenant_session` will add) and it provisions nothing: no dedicated database exists anywhere in
this template today. It is, unmodified, the exact seam a later tenant-lifecycle tool calls into
when it actually provisions or erases a dedicated tenant's database, and the exact seam
`tenant_session` will route through once it starts asking the control plane which alias a tenant
is assigned. The documented triggers for actually provisioning the first dedicated tenant --
when it becomes worth exercising the path this module keeps ready -- are ADR-0002's own
"Revisit when" list (`docs/adr/0002-hybrid-tenant-isolation.md`); they are not repeated here so
the two can never drift apart.

Every dedicated engine is guarded (issue #81, ADR-0002) the moment it is first built, inside the
same per-alias lock that serializes concurrent first requests: `app.db.guard.check_role_and_rls`
runs against a connection from the brand-new engine before it is ever cached in `_engines` or
handed back to a caller. A failing check disposes the engine, never caches it, and propagates --
the request that triggered the first open fails closed, and the next call rebuilds and rechecks
from scratch rather than serving anything from a bad cache entry. This closes the gap where a
dedicated alias, opened lazily by a real request, would otherwise go unchecked until the next
`run_role_rls_guard()`/`/ready` cycle. The pooled alias is unaffected: it is guarded the same way
it always was, by `run_role_rls_guard`/`/ready`, never by this module.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config import get_settings
from app.db.session import _reset_session_state, get_engine

# Reserved alias for the one engine that exists in every deployment, pooled or not: the shared
# database built from Settings.database_url. Never looked up via a secret file.
POOLED_ALIAS = "pooled"

# Directory holding one tenant-secret file per dedicated database alias (ADR-0011): the file
# named exactly <alias> holds that alias's full connection string (an asyncpg DSN -- host, port,
# database, user, and password, since a dedicated database is not assumed to share a host with
# the pooled one), delivered by the deployment, never a field on Settings or tenants.settings.
# Overridable via TENANT_DB_SECRETS_DIR (tests use this; production mounts the real directory).
_DEFAULT_TENANT_DB_SECRETS_DIR = "/run/secrets/tenant-db"


def _secrets_dir() -> Path:
    return Path(os.environ.get("TENANT_DB_SECRETS_DIR", _DEFAULT_TENANT_DB_SECRETS_DIR))


class UnknownDatabaseAliasError(RuntimeError):
    """Raised when a database alias has no matching tenant-secret file.

    This is a fail-closed error, not a fallback: a caller asking for an alias the deployment has
    not actually provisioned a secret for gets a clear, documented failure instead of silently
    being routed to the pooled engine and (mis)reaching the wrong tenants' data.
    """

    def __init__(self, alias: str, path: Path) -> None:
        super().__init__(
            f"No tenant-secret file for database alias {alias!r} at {path} -- refusing to fall "
            "back to the pooled engine. Provision the secret file (or provision the alias "
            "through the tenant-lifecycle tool) before routing a tenant to it."
        )
        self.alias = alias


# Process-wide cache: alias -> the AsyncEngine built for it. Dedicated aliases only -- the
# pooled alias is served straight from app.db.session.get_engine() and never stored here.
_engines: dict[str, AsyncEngine] = {}

# One asyncio.Lock per alias currently being (or about to be) built, so concurrent first
# requests for the same not-yet-cached alias serialize on the *same* lock instead of each
# building their own engine. Populating this dict itself never awaits between the `.get` and the
# `.setdefault` below, so it stays race-free even though asyncio.Lock isn't itself atomic.
_locks: dict[str, asyncio.Lock] = {}


def _lock_for(alias: str) -> asyncio.Lock:
    lock = _locks.get(alias)
    if lock is None:
        lock = _locks.setdefault(alias, asyncio.Lock())
    return lock


def _read_dsn(alias: str) -> str:
    path = _secrets_dir() / alias
    try:
        dsn = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise UnknownDatabaseAliasError(alias, path) from None
    if not dsn:
        raise UnknownDatabaseAliasError(alias, path)
    return dsn


def _build_engine(dsn: str) -> AsyncEngine:
    """Build one AsyncEngine for a dedicated alias, with the same pool shape and the same
    session-local-state checkin guard as the pooled engine (app/db/session.py)."""
    settings = get_settings()
    engine = create_async_engine(
        dsn,
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_recycle=settings.db_pool_recycle,
    )
    event.listen(engine.sync_engine, "checkin", _reset_session_state)
    return engine


async def get_engine_for_alias(alias: str) -> AsyncEngine:
    """Return the engine that serves `alias`, building and caching it if this is the first
    request for it. The pooled alias is pre-registered (via app.db.session.get_engine()) and
    never reads a secret file; any other alias is built once, lazily, from that alias's
    tenant-secret file and cached for the life of the process.

    A brand-new dedicated engine is guarded (`app.db.guard.check_role_and_rls`, issue #81) before
    it is cached or returned: a connected superuser/BYPASSRLS role, or a public-schema table
    missing forced Row-Level Security, disposes the engine and raises
    `PrivilegedRoleOrMissingRLSError` instead of caching or handing back a bad engine. The next
    call for the same alias starts over -- re-reading the secret file, rebuilding the engine, and
    re-running the check -- rather than ever serving a cached engine that failed its check. The
    pooled alias is never checked here; its own guarding is `run_role_rls_guard`/`/ready`,
    unchanged."""
    if alias == POOLED_ALIAS:
        return get_engine()

    cached = _engines.get(alias)
    if cached is not None:
        return cached

    async with _lock_for(alias):
        # Re-check inside the lock: a concurrent caller may have just finished building it.
        cached = _engines.get(alias)
        if cached is not None:
            return cached
        dsn = _read_dsn(alias)
        engine = _build_engine(dsn)
        # Guard before caching or returning (issue #81): imported lazily to avoid a module-level
        # import cycle with app.db.guard, which itself imports this module's
        # get_engine_for_alias lazily inside its own functions.
        from app.db.guard import check_role_and_rls

        try:
            async with engine.connect() as conn:
                await check_role_and_rls(conn)
        except Exception:
            await engine.dispose()
            raise
        _engines[alias] = engine
        return engine


def reset_registry_for_tests() -> None:
    """Test-only: drop every cached dedicated engine and its lock, so the next request for an
    alias rebuilds it (and re-reads its secret file) from scratch. Never call this in the
    running application -- it does not dispose the engines it drops."""
    _engines.clear()
    _locks.clear()
