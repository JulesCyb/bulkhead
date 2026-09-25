"""Fail-closed role/RLS guard (ADR-0011, issue #15).

The application refuses to ever start, and refuses to ever report itself ready, while it is
connected as a database superuser, as a role that bypasses Row-Level Security, or while any
table in the `public` schema lacks forced Row-Level Security. This is the same live check run
once at ASGI lifespan startup (a failure there prevents the process from ever accepting traffic)
and again, live, on every call to the `/ready` endpoint — never cached from the boot-time result.

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


async def run_role_rls_guard() -> None:
    """Opens a connection from the application's own engine and runs `check_role_and_rls` on
    it. This is the entry point both the ASGI lifespan (startup) and the readiness endpoint
    (every call) use, so there is exactly one live implementation of the guard, not two that
    could drift apart."""
    from app.db.session import get_engine

    engine = get_engine()
    async with engine.connect() as conn:
        await check_role_and_rls(conn)
