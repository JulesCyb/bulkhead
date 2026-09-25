"""The operator CLI (Spec 9 / #68, ADR-0010): a runnable command-line entry point
(`scripts/operator.py` is the thin wrapper) connecting only as the owner database role.

Every invocation reads its connection string from `app.migration_settings.get_migration_settings`
-- the same owner DSN Alembic's migrations use -- and nothing else: there is no flag or
environment variable here that accepts a different connection string, so this module can never
fall back to, or be pointed at, the cluster superuser. `app.config.Settings` (the long-running
API's settings object) is never imported here either -- `create` (Spec 9 / #70) needs it (model
allow-list validation, gateway defaults), so that work lives in `app.operator.create`, imported
by name here rather than reached through `app.config` directly.

Command dispatch and audit recording are deliberately two separate transactions on two separate
connections (see `_run` below): if a command's own transaction rolls back on failure, the audit
row describing that failure must still commit.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from app.migration_settings import get_migration_settings
from app.operator.audit import UNSCOPED_TENANT_ID, record_action
from app.operator.create import create_tenant
from app.operator.listing import list_tenants

OPERATOR_IDENTITY_ENV_VAR = "OPERATOR_IDENTITY"


def _operator_identity() -> str:
    """Who is running this command, for the audit log. Never guessed from the database
    connection (that is always `app_owner`, the same for every operator) -- from the
    environment, defaulting to the OS user only as a local-development convenience."""
    return os.environ.get(OPERATOR_IDENTITY_ENV_VAR) or os.environ.get("USER", "unknown")


async def _run_list(conn: AsyncConnection, args: argparse.Namespace) -> str:
    tenants = await list_tenants(conn)
    if not tenants:
        print("No tenants in the control plane.")
    for t in tenants:
        print(
            f"{t.tenant_id}  {t.name!r:30}  tier={t.isolation_tier:9}  "
            f"residency={t.residency or '-':6}  alias={t.database_alias or '-':12}  "
            f"suspended={t.suspended}"
        )
    return f"ok: listed {len(tenants)} tenant(s)"


async def _run_create(conn: AsyncConnection, args: argparse.Namespace) -> str:
    result = await create_tenant(
        conn,
        tenant_name=args.tenant_name,
        residency=args.residency,
        admin_email=args.admin_email,
        model=args.model,
        issuer=args.issuer,
        subject=args.subject,
    )
    print(f"MCP_TENANT_ID={result.tenant_id}")
    print(f"MCP_IDENTITY_ID={result.identity_id}")
    print(f"Gateway credential alias: {result.gateway_credential_alias}")
    print(
        f"control-plane record: {result.control_plane}; "
        f"gateway credential: {result.gateway_credential}; "
        f"admin membership: {result.admin_membership}"
    )
    print(
        f"\ncurl -H 'X-Identity-Id: {result.identity_id}' "
        f"http://localhost:8000/v1/t/{result.tenant_id}/agents/assistant/run ..."
    )
    return (
        f"ok: tenant {result.tenant_id} "
        f"(control-plane: {result.control_plane}, "
        f"gateway: {result.gateway_credential}, "
        f"membership: {result.admin_membership})"
    )


# Every command's target tenant id for the audit log. `list` has none -- it targets every
# tenant, not one -- so it uses the documented sentinel (see app.operator.audit). `create`
# resolves or mints its own tenant id as part of the command itself, so it uses the same
# sentinel here too rather than threading a second return value through this dispatch table.
_COMMANDS = {
    "list": _run_list,
    "create": _run_create,
}


async def _run(command: str, args: argparse.Namespace) -> int:
    dsn = get_migration_settings().database_url_migrations.get_secret_value()
    engine = create_async_engine(dsn)
    started_at = datetime.now(UTC)
    outcome = "error"
    error: str | None = None
    exit_code = 1
    try:
        async with engine.begin() as conn:
            outcome = await _COMMANDS[command](conn, args)
        exit_code = 0
    except Exception as exc:  # noqa: BLE001 - recorded below, then re-raised as a nonzero exit
        error = f"{type(exc).__name__}: {exc}"
        outcome = "error"
    finally:
        finished_at = datetime.now(UTC)
        async with engine.begin() as audit_conn:
            await record_action(
                audit_conn,
                operator=_operator_identity(),
                command=command,
                target_tenant_id=UNSCOPED_TENANT_ID,
                args=vars(args),
                outcome=outcome,
                started_at=started_at,
                finished_at=finished_at,
                error=error,
            )
        await engine.dispose()
    if error is not None:
        print(f"error: {error}")
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="operator",
        description="Operator tool: audited commands run as the owner database role.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "list",
        help="List every tenant's isolation tier, residency, database alias, and suspension state.",
    )

    create_parser = sub.add_parser(
        "create",
        help="Create (idempotently) a pooled tenant: control-plane record, gateway credential, "
        "and first admin membership. See app.operator.create for the full contract.",
    )
    create_parser.add_argument("tenant_name")
    create_parser.add_argument(
        "--residency",
        required=True,
        help="e.g. eu, us -- validated against this deployment's residency allow-list before "
        "anything is written",
    )
    create_parser.add_argument(
        "--admin-email", dest="admin_email", required=True, help="the first admin's email"
    )
    create_parser.add_argument(
        "--model",
        default=None,
        help="optional tenants.settings['model'] override, validated against the residency's "
        "model allow-list before anything is written",
    )
    create_parser.add_argument(
        "--issuer",
        default="dev-seed",
        help="identity issuer to match/create for the admin identity (default: dev-seed)",
    )
    create_parser.add_argument(
        "--subject",
        default=None,
        help="identity subject to match/create (default: the admin email)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(_run(args.command, args))


if __name__ == "__main__":
    raise SystemExit(main())
