"""The tenant-table registry (ADR-0010, Spec 9 / #65).

`TENANT_TABLES` is the single, explicit list of every table *currently* in the tenant-editable
(`public`) schema that carries a `tenant_id` column — the present-day state at the migration
head, not a historical record. Two things depend on it staying complete:

- Each migration that creates or retires a registered table applies (or, on retirement, drops)
  Row-Level Security for exactly the tables it itself creates, using its own frozen snapshot of
  those names — never by importing this live module, which would apply RLS to a table that does
  not exist yet when an old migration is replayed on a fresh database (issue #23 learned this the
  hard way: 0009 retired `users` from this registry in favor of `memberships`, a table
  `migrations/versions/0001_initial.py` never creates).
- The tenant lifecycle tool (Spec 9) will iterate it to reach every tenant's rows on erasure.

`unregistered_tenant_tables()` is the other half: it introspects the real schema and returns
every `public` table with a `tenant_id` column that this registry does not list, so a table
added without registering it here fails a check immediately instead of silently escaping RLS
and, later, erasure.

Scope is `public` only, by construction of the query below (`table_schema = 'public'`) — the
operator-only `control` schema (0002_control_plane_schema.py) is never introspected, so a
control-plane table that keys by `tenant_id` without a cascading foreign key to `public.tenants`
is not required to register here.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

TENANT_TABLES: tuple[str, ...] = (
    "memberships",
    "documents",
    "agent_credentials",
    "conversations",
    "messages",
    "pending_actions",
    "standing_grants",
)


async def unregistered_tenant_tables(conn: AsyncConnection) -> list[str]:
    """Every table in the `public` schema with a `tenant_id` column that is not registered
    in `TENANT_TABLES`. Empty when the registry has full coverage."""
    result = await conn.execute(
        text(
            """
            SELECT table_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND column_name = 'tenant_id'
            ORDER BY table_name
            """
        )
    )
    return [table for (table,) in result if table not in TENANT_TABLES]
