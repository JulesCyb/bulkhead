"""Creates the first tenant and user and prints the IDs for .env / dev headers.

Runs with DATABASE_URL_MIGRATIONS and sets the tenant context before writing: FORCE ROW LEVEL
SECURITY binds even the table owner (only superusers bypass RLS), so on managed Postgres
(RDS/Neon/Supabase) the owner role would otherwise be blocked by the policies.

    uv run python scripts/seed.py "My Tenant" me@example.com
"""

from __future__ import annotations

import asyncio
import sys
import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.migration_settings import get_migration_settings


async def main(tenant_name: str, email: str) -> None:
    dsn = get_migration_settings().database_url_migrations.get_secret_value()
    engine = create_async_engine(dsn)
    tenant_id, user_id = uuid.uuid4(), uuid.uuid4()
    async with engine.begin() as conn:
        # Satisfies the policies' WITH CHECK even when the role is owner-but-not-superuser.
        await conn.execute(
            text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": str(tenant_id)}
        )
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant_id, "name": tenant_name},
        )
        await conn.execute(
            text(
                "INSERT INTO users (id, tenant_id, email, role) "
                "VALUES (:id, :tenant_id, :email, 'admin')"
            ),
            {"id": user_id, "tenant_id": tenant_id, "email": email},
        )
    await engine.dispose()
    print(f"MCP_TENANT_ID={tenant_id}\nMCP_IDENTITY_ID={user_id}")
    print(f"\ncurl -H 'X-Tenant-Id: {tenant_id}' -H 'X-Identity-Id: {user_id}' ...")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python scripts/seed.py <tenant name> <email>")
    asyncio.run(main(sys.argv[1], sys.argv[2]))
