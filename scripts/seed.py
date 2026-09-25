"""Creates the first tenant, a global identity, and an admin membership binding them, mints the
tenant a gateway credential, and prints the IDs for .env / dev headers.

Runs with DATABASE_URL_MIGRATIONS and sets the tenant context before writing: FORCE ROW LEVEL
SECURITY binds even the table owner (only superusers bypass RLS), so on managed Postgres
(RDS/Neon/Supabase) the owner role would otherwise be blocked by the policies. `control.identities`
carries no tenant_id and no RLS (ADR-0003), so that insert needs no tenant context at all --
issued before the tenant context is set, alongside the tenant insert.

Also calls `app.gateway_provisioning.provision_gateway_credential` (Spec 7 / #53) directly --
there is no operator command for this yet (Spec 9 builds one) -- so the documented quickstart
keeps producing a tenant that can actually call the assistant, cost-isolated and confined to its
residency from its first request. `admin_client` is exposed only so tests can inject a fake
gateway; the real CLI path always builds one from `LITELLM_BASE_URL`/`LITELLM_MASTER_KEY`.

    uv run python scripts/seed.py "My Tenant" me@example.com
"""

from __future__ import annotations

import asyncio
import sys
import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import Settings, get_settings
from app.gateway_provisioning import (
    GatewayAdminClient,
    GatewayCredentialLimits,
    provision_gateway_credential,
)
from app.migration_settings import get_migration_settings


async def main(
    tenant_name: str,
    email: str,
    *,
    settings: Settings | None = None,
    admin_client: GatewayAdminClient | None = None,
) -> None:
    settings = settings or get_settings()
    dsn = get_migration_settings().database_url_migrations.get_secret_value()
    engine = create_async_engine(dsn)
    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    async with engine.begin() as conn:
        # Global identity: no tenant_id column, no RLS (ADR-0003) -- issued before any tenant
        # context is set. "dev-seed" is a placeholder issuer for AUTH_MODE=dev-headers; a real
        # identity provider replaces it once AUTH_MODE=jwt is implemented.
        await conn.execute(
            text(
                "INSERT INTO control.identities (id, issuer, subject, display_name, email) "
                "VALUES (:id, 'dev-seed', :email, :email, :email)"
            ),
            {"id": identity_id, "email": email},
        )
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
                "INSERT INTO memberships (tenant_id, identity_id, role) "
                "VALUES (:tenant_id, :identity_id, 'admin')"
            ),
            {"tenant_id": tenant_id, "identity_id": identity_id},
        )

    alias = await provision_gateway_credential(
        tenant_id,
        residency=settings.residency,
        limits=GatewayCredentialLimits(
            spend_ceiling_usd=settings.gateway_default_spend_ceiling_usd,
            budget_reset_period=settings.gateway_default_budget_reset_period,
            requests_per_minute=settings.gateway_default_requests_per_minute,
            tokens_per_minute=settings.gateway_default_tokens_per_minute,
        ),
        settings=settings,
        admin_client=admin_client,
        owner_engine=engine,
    )
    await engine.dispose()
    print(f"MCP_TENANT_ID={tenant_id}\nMCP_IDENTITY_ID={identity_id}")
    print(f"Gateway credential alias: {alias}")
    print(
        f"\ncurl -H 'X-Identity-Id: {identity_id}' "
        f"http://localhost:8000/v1/t/{tenant_id}/agents/assistant/run ..."
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python scripts/seed.py <tenant name> <email>")
    asyncio.run(main(sys.argv[1], sys.argv[2]))
