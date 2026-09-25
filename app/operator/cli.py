"""The operator CLI (Spec 9 / #68, ADR-0010): a runnable command-line entry point
(`scripts/operator.py` is the thin wrapper) connecting only as the owner database role.

Every invocation reads its connection string from `app.migration_settings.get_migration_settings`
-- the same owner DSN migrations and `scripts/seed.py` use -- and nothing else: there is no flag
or environment variable here that accepts a different connection string, so this module can never
fall back to, or be pointed at, the cluster superuser. `app.config.Settings` (the long-running
API's settings object) is never imported here either.

Command dispatch and audit recording are deliberately two separate transactions on two separate
connections (see `_run` below): if a command's own transaction rolls back on failure, the audit
row describing that failure must still commit.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from app.migration_settings import get_migration_settings
from app.operator.audit import UNSCOPED_TENANT_ID, record_action
from app.operator.listing import list_tenants
from app.operator.suspend import set_tenant_suspended

OPERATOR_IDENTITY_ENV_VAR = "OPERATOR_IDENTITY"


def _operator_identity() -> str:
    """Who is running this command, for the audit log. Never guessed from the database
    connection (that is always `app_owner`, the same for every operator) -- from the
    environment, defaulting to the OS user only as a local-development convenience."""
    return os.environ.get(OPERATOR_IDENTITY_ENV_VAR) or os.environ.get("USER", "unknown")


async def _run_list(conn: AsyncConnection, args: argparse.Namespace) -> tuple[str, None]:
    tenants = await list_tenants(conn)
    if not tenants:
        print("No tenants in the control plane.")
    for t in tenants:
        print(
            f"{t.tenant_id}  {t.name!r:30}  tier={t.isolation_tier:9}  "
            f"residency={t.residency or '-':6}  alias={t.database_alias or '-':12}  "
            f"suspended={t.suspended}"
        )
    # `list` has no single target tenant -- it targets every tenant -- so its audit row uses the
    # documented sentinel (see app.operator.audit), not a real id.
    return f"ok: listed {len(tenants)} tenant(s)", None


async def _run_suspend_or_unsuspend(
    conn: AsyncConnection, args: argparse.Namespace, *, suspended: bool
) -> tuple[str, UUID]:
    verb = "suspended" if suspended else "unsuspended"
    result = await set_tenant_suspended(conn, args.identifier, suspended=suspended)
    if not result.changed:
        outcome = f"no-op: {result.name!r} ({result.tenant_id}) was already {verb}"
    else:
        outcome = f"ok: {result.name!r} ({result.tenant_id}) is now {verb}"
    print(outcome)
    return outcome, result.tenant_id


async def _run_suspend(conn: AsyncConnection, args: argparse.Namespace) -> tuple[str, UUID]:
    return await _run_suspend_or_unsuspend(conn, args, suspended=True)


async def _run_unsuspend(conn: AsyncConnection, args: argparse.Namespace) -> tuple[str, UUID]:
    return await _run_suspend_or_unsuspend(conn, args, suspended=False)


# Every command's target tenant id for the audit log. `list` has none -- it targets every
# tenant, not one -- so it uses the documented sentinel (see app.operator.audit).
_COMMANDS = {
    "list": _run_list,
    "suspend": _run_suspend,
    "unsuspend": _run_unsuspend,
}


async def _run(command: str, args: argparse.Namespace) -> int:
    dsn = get_migration_settings().database_url_migrations.get_secret_value()
    engine = create_async_engine(dsn)
    started_at = datetime.now(UTC)
    outcome = "error"
    target_tenant_id: UUID | None = None
    error: str | None = None
    exit_code = 1
    try:
        async with engine.begin() as conn:
            outcome, target_tenant_id = await _COMMANDS[command](conn, args)
        exit_code = 0
    except Exception as exc:  # noqa: BLE001 - recorded below, then re-raised as a nonzero exit
        # A lookup failure (TenantNotFoundError/AmbiguousTenantNameError) never resolved a real
        # tenant, so `target_tenant_id` stays None here -- the sentinel below is the only honest
        # target id for this row, same as any other failure.
        error = f"{type(exc).__name__}: {exc}"
        outcome = "error"
    finally:
        finished_at = datetime.now(UTC)
        async with engine.begin() as audit_conn:
            await record_action(
                audit_conn,
                operator=_operator_identity(),
                command=command,
                target_tenant_id=(
                    target_tenant_id if target_tenant_id is not None else UNSCOPED_TENANT_ID
                ),
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
    suspend_parser = sub.add_parser(
        "suspend",
        help="Suspend a tenant (id or unambiguous name); a no-op if already suspended.",
    )
    suspend_parser.add_argument("identifier", help="tenant id or unambiguous name")
    unsuspend_parser = sub.add_parser(
        "unsuspend",
        help="Un-suspend a tenant (id or unambiguous name); a no-op if already active.",
    )
    unsuspend_parser.add_argument("identifier", help="tenant id or unambiguous name")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(_run(args.command, args))


if __name__ == "__main__":
    raise SystemExit(main())
