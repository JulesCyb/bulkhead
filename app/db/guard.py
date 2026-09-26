"""Fail-closed role/RLS guard (ADR-0011, issue #15; extended to every open engine by Spec 10 /
#77).

The application refuses to ever start, and refuses to ever report itself ready, while any
database engine it might open is connected as a database superuser, as a role that bypasses Row-
Level Security, or while any table in the `public` schema of that engine's database lacks forced
Row-Level Security. This is the same live check run once at ASGI lifespan startup (a failure
there prevents the process from ever accepting traffic) and again, live, on every call to the
`/ready` endpoint — never cached from the boot-time result.

`run_role_rls_guard` no longer checks only the one pooled `DATABASE_URL`: it first asks the control
plane (`control.enumerate_database_aliases()`, via `DatabaseAliasRepository`) which database aliases
are currently referenced -- the pooled default, always, plus every dedicated alias at least one
tenant is assigned to (ADR-0002's hybrid-isolation seam, `app/db/engine_registry.py`) -- and then
runs the identical check against each one's engine in turn. A dangling or misconfigured
dedicated database (a privileged role, or a table missing forced RLS) fails startup exactly as a
misconfigured pooled database would, even though nothing has routed a real request to it yet;
iterating every alias, rather than stopping at the first one checked, is the point of the
extension.

Reuses `app.db.models.TENANT_ISOLATION_EXCEPTIONS`, the same exception list the schema-invariant
test (`tests/test_rls_integration.py::test_public_schema_tenant_isolation_invariant`) enforces,
so a legitimately tenant-less table is a one-line addition to a single list, never a second one
that can silently drift apart from it.
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.db.models import TENANT_ISOLATION_EXCEPTIONS

log = logging.getLogger(__name__)


# Role-level settings for the `app` role (Spec 7 / #55; relocated next to this guard, the
# module that actually cares about the `app` role's privileges, by spec A4 / #94, #112): applied
# once in docker/postgres/01-init.sh, mirrored here so the embedded-Postgres integration test can
# assert them without duplicating literals. Independent of `Settings.db_statement_timeout_ms`
# (`app/config.py`), which is the per-transaction timeout the application sets on every
# tenant_session().
ROLE_STATEMENT_TIMEOUT_MS = 60_000
ROLE_CONNECTION_LIMIT = 50


class PrivilegedRoleOrMissingRLSError(RuntimeError):
    """The connected role is privileged (superuser/BYPASSRLS), or a table in the `public` schema
    lacks forced Row-Level Security. The message never names the offending role or table — those
    are logged server-side only; a caller (e.g. the readiness endpoint) must only ever see a
    generic failure, never a detail that could be used to fingerprint the deployment."""


async def check_role_and_rls(conn: AsyncConnection) -> None:
    """Live query against `pg_roles`/`pg_class`/`pg_namespace` on the given connection. Re-run on
    every call — no result is ever cached across calls. Raises
    ``PrivilegedRoleOrMissingRLSError`` if the connected role is a superuser or holds
    `BYPASSRLS`, or if any table in `public` outside `TENANT_ISOLATION_EXCEPTIONS` lacks forced
    Row-Level Security."""
    role_row = (
        await conn.execute(
            text(
                "SELECT rolname, rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
            )
        )
    ).one()
    if role_row.rolsuper or role_row.rolbypassrls:
        log.error(
            "Fail-closed guard: connected role %r is a superuser or holds BYPASSRLS — "
            "refusing to run.",
            role_row.rolname,
        )
        raise PrivilegedRoleOrMissingRLSError(
            "The database connection is privileged (superuser or BYPASSRLS) — refusing to run."
        )

    unforced = (
        await conn.execute(
            text(
                "SELECT c.relname FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = 'public' AND c.relkind = 'r' AND NOT c.relforcerowsecurity"
            )
        )
    ).all()
    offending = [row.relname for row in unforced if row.relname not in TENANT_ISOLATION_EXCEPTIONS]
    if offending:
        # Count only, to the process log — never the table names, and never to a caller.
        log.error(
            "Fail-closed guard: %d table(s) in the public schema lack forced Row-Level "
            "Security — refusing to run.",
            len(offending),
        )
        raise PrivilegedRoleOrMissingRLSError(
            "One or more tables lack forced Row-Level Security — refusing to run."
        )


async def _referenced_aliases() -> list[str]:
    """Every database alias the guard must check: the pooled default, unconditionally, plus
    whatever `control.enumerate_database_aliases()` reports (the pooled default again, mirrored
    back for every pooled tenant, plus one entry per distinct dedicated alias). The pooled alias
    is included even before any tenant has been seeded, so a deployment with an empty control
    plane still guards the one engine it actually opens."""
    from app.db.engine_registry import POOLED_ALIAS
    from app.db.session import control_session
    from app.repositories.control import DatabaseAliasRepository

    async with control_session() as session:
        referenced = await DatabaseAliasRepository().list_referenced_aliases(session)
    return sorted({POOLED_ALIAS, *referenced})


async def run_role_rls_guard() -> None:
    """Runs `check_role_and_rls` against every database engine the process might open: the
    pooled one always, plus every dedicated alias `control.enumerate_database_aliases()` currently
    references. Each alias's engine comes from `app.db.engine_registry.get_engine_for_alias`
    (built lazily and cached, or `UnknownDatabaseAliasError` if a referenced alias has no
    matching secret file -- itself a fail-closed condition this function does not catch). This
    is the entry point both the ASGI lifespan (startup) and the readiness endpoint (every call)
    use, so there is exactly one live implementation of the guard, not two that could drift
    apart."""
    from app.db.engine_registry import get_engine_for_alias

    for alias in await _referenced_aliases():
        engine = await get_engine_for_alias(alias)
        async with engine.connect() as conn:
            await check_role_and_rls(conn)
