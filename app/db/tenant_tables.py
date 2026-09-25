"""The tenant-table registry (ADR-0010, Spec 9 / #65).

`TENANT_TABLES` is the single, explicit list of every table in the tenant-editable (`public`)
schema that carries a `tenant_id` column. Two things depend on it staying complete:

- Migration tooling iterates it to apply Row-Level Security (see
  `migrations/versions/0001_initial.py`) instead of keeping a private, per-migration list.
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

TENANT_TABLES: tuple[str, ...] = ("users", "documents")


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
