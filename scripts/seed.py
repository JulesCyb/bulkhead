"""Two modes, both connected with the owner-role migrations connection (DATABASE_URL_MIGRATIONS):

**Default (`seed`) mode** creates the first tenant, a global identity, and an admin membership
binding them, mints the tenant a gateway credential, and prints the IDs for .env / dev headers::

    uv run python scripts/seed.py "My Tenant" me@example.com

**`add-membership` mode** attaches a second membership -- any of the four roles CONTEXT.md
defines (`admin`, `member`, `support`, `agent`) -- to an *already-seeded* tenant, for a new or an
existing identity, without touching that tenant's first membership. This is the documented,
owner-role stand-in for an admin-facing invitation flow (not built yet): it lets a developer
exercise a consultant's second membership or an operator's support access locally::

    uv run python scripts/seed.py add-membership <tenant-id> support consultant@example.com
    uv run python scripts/seed.py add-membership <tenant-id> member consultant@example.com

    # Reuses the identity already created above (same issuer+subject, matched by email as the
    # subject by default) rather than creating a second one:
    uv run python scripts/seed.py add-membership <other-tenant-id> member consultant@example.com

    # --issuer/--subject override the identity lookup key directly, e.g. to attach a membership
    # to an identity that was not seeded with the default dev-seed issuer:
    uv run python scripts/seed.py add-membership <tenant-id> agent svc@example.com \\
        --issuer dev-seed --subject svc-account-42

Both modes set the tenant context (`set_config('app.tenant_id', ...)`) before writing to
`memberships`: FORCE ROW LEVEL SECURITY binds even the table owner (only superusers bypass RLS),
so on managed Postgres (RDS/Neon/Supabase) the owner role would otherwise be blocked by the
policies. `control.identities` carries no tenant_id and no RLS (ADR-0003), so identity writes need
no tenant context at all.

Default mode also calls `app.gateway_provisioning.provision_gateway_credential` (Spec 7 / #53)
directly -- there is no operator command for this yet (Spec 9 builds one) -- so the documented
quickstart keeps producing a tenant that can actually call the assistant, cost-isolated and
confined to its residency from its first request. `admin_client` is exposed only so tests can
inject a fake gateway; the real CLI path always builds one from
`LITELLM_BASE_URL`/`LITELLM_MASTER_KEY`. `add-membership` mode never touches the gateway -- the
tenant was already provisioned by the `seed` run that created it.
"""

from __future__ import annotations

import argparse
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

#: The four membership roles CONTEXT.md defines. `memberships.role` also carries a DB CHECK
#: constraint restricted to these (migration 0009); this set lets the CLI fail fast, before ever
#: opening a connection, on a typoed role.
VALID_ROLES = frozenset({"admin", "member", "support", "agent"})


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


async def add_membership(
    tenant_id: uuid.UUID,
    role: str,
    email: str,
    *,
    issuer: str = "dev-seed",
    subject: str | None = None,
) -> uuid.UUID:
    """Attaches a second membership to an already-seeded tenant (#25): for a new or an existing
    identity, in any of the four defined roles, leaving the tenant's first membership -- and
    every other membership -- completely untouched. Never provisions a gateway credential; the
    tenant was already provisioned when it was first seeded.

    The identity is matched by `(issuer, subject)` -- `control.identities`'s own unique key --
    with `subject` defaulting to `email` so that running this twice with the same email reuses
    the identity (a genuine second membership for one person) instead of creating a duplicate.
    Pass `--subject` explicitly to attach a membership to an identity seeded under a different
    subject than its email.
    """
    if role not in VALID_ROLES:
        raise ValueError(f"role must be one of {sorted(VALID_ROLES)}, got {role!r}")
    subject = subject or email

    dsn = get_migration_settings().database_url_migrations.get_secret_value()
    engine = create_async_engine(dsn)
    async with engine.begin() as conn:
        # Find-or-create by (issuer, subject): a fresh id is proposed, but the RETURNING id is
        # the existing row's when one already matches -- ON CONFLICT DO UPDATE (rather than DO
        # NOTHING) is what makes RETURNING fire on that path too.
        identity_id = (
            await conn.execute(
                text(
                    "INSERT INTO control.identities (id, issuer, subject, display_name, email) "
                    "VALUES (:id, :issuer, :subject, :email, :email) "
                    "ON CONFLICT (issuer, subject) DO UPDATE SET issuer = EXCLUDED.issuer "
                    "RETURNING id"
                ),
                {"id": uuid.uuid4(), "issuer": issuer, "subject": subject, "email": email},
            )
        ).scalar_one()
        # Satisfies the memberships policy's WITH CHECK even when the role is owner-but-not-
        # superuser (FORCE ROW LEVEL SECURITY), same as `main` above.
        await conn.execute(
            text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": str(tenant_id)}
        )
        await conn.execute(
            text(
                "INSERT INTO memberships (tenant_id, identity_id, role) "
                "VALUES (:tenant_id, :identity_id, :role)"
            ),
            {"tenant_id": tenant_id, "identity_id": identity_id, "role": role},
        )
    await engine.dispose()

    print(f"MCP_IDENTITY_ID={identity_id}")
    print(f"Membership: role={role} tenant={tenant_id}")
    print(
        f"\ncurl -H 'X-Identity-Id: {identity_id}' "
        f"http://localhost:8000/v1/t/{tenant_id}/agents/assistant/run ..."
    )
    return identity_id


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scripts/seed.py",
        description="Seed a fresh tenant admin, or attach a second membership to an existing "
        "tenant.",
    )
    subparsers = parser.add_subparsers(dest="mode")

    seed_parser = subparsers.add_parser(
        "seed",
        help="default: create a tenant, an admin identity, and its admin membership",
    )
    seed_parser.add_argument("tenant_name")
    seed_parser.add_argument("email")

    add_parser = subparsers.add_parser(
        "add-membership",
        help="attach another membership (any of the four roles) to an already-seeded tenant",
    )
    add_parser.add_argument("tenant_id", type=uuid.UUID)
    add_parser.add_argument("role", choices=sorted(VALID_ROLES))
    add_parser.add_argument("email")
    add_parser.add_argument(
        "--issuer", default="dev-seed", help="identity issuer to match/create (default: dev-seed)"
    )
    add_parser.add_argument(
        "--subject", default=None, help="identity subject to match/create (default: the email)"
    )

    # `uv run python scripts/seed.py "My Tenant" me@example.com` -- default mode with no
    # subcommand named, exactly as before this ticket.
    if argv and argv[0] not in ("seed", "add-membership", "-h", "--help"):
        argv = ["seed", *argv]

    args = parser.parse_args(argv)
    if args.mode is None:
        parser.print_usage(sys.stderr)
        sys.exit(2)
    return args


if __name__ == "__main__":
    parsed = _parse_args(sys.argv[1:])
    if parsed.mode == "add-membership":
        asyncio.run(
            add_membership(
                parsed.tenant_id,
                parsed.role,
                parsed.email,
                issuer=parsed.issuer,
                subject=parsed.subject,
            )
        )
    else:
        asyncio.run(main(parsed.tenant_name, parsed.email))
