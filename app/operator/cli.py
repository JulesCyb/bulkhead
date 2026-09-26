"""The operator CLI (Spec 9 / #68, ADR-0010; public entry point spec A5 / #115): a runnable
command-line entry point (`scripts/operator.py` is the thin wrapper) connecting only as the owner
database role.

`main(argv, *, engine=None, admin_client=None)` is the one public function here -- it parses
`argv`, dispatches to the matching command, records the operator-action audit row, prints the
result, and returns the process exit code. Dispatch (`_COMMANDS`, `_run`) stays private: a test
drives a command end to end through `main()` alone, with an injected `engine` and (for `create`/
`erase`) an injected `admin_client`, never by importing `_run`/`build_parser`/`_COMMANDS`
directly. `scripts/operator.py` calls `main()` unchanged.

`main()` always owns whichever engine it ends up using -- one it builds itself from the owner DSN
(`app.db.lifecycle.owner_engine`) when none is given, or one a caller injects -- and disposes it
before returning, on whichever event loop actually used it (see below): a caller that injects an
engine hands over its lifecycle for exactly the one call, the same way `owner_engine` already
does for the DSN case, rather than the two cases leaving disposal split across two different
owners. `main()` is also callable from inside a running event loop (an async test driving it
directly, most commonly): `asyncio.run()` cannot itself be called there, so in that case the
whole call runs to completion on a fresh loop of its own, in one worker thread, rather than
raising -- this is also why an injected engine's connections must never be opened before this
call (they would be bound to the wrong loop): build it and hand it straight to `main()`.

Every invocation reads its connection string from `app.migration_settings.get_migration_settings`
-- the same owner DSN Alembic's migrations use -- and nothing else, unless a caller injects an
`engine` itself: there is no flag or environment variable here that accepts a different
connection string, so this module can never fall back to, or be pointed at, the cluster
superuser. `app.config.Settings` (the long-running API's settings object) is never imported here
either -- `create` (Spec 9 / #70) needs it (model allow-list validation, gateway defaults), so
that work lives in `app.operator.create`, imported by name here rather than reached through
`app.config` directly.

Command dispatch and audit recording are deliberately two separate transactions on the same
engine (see `_run` below): if a command's own transaction rolls back on failure, the audit row
describing that failure must still commit.

One formatter per command, not five: every command's result type (`app.operator.listing.
TenantListing`, `app.operator.suspend.SuspendResult`, `app.operator.create.CreateTenantResult`,
`app.operator.erase.EraseResult`) implements the same small protocol -- `render()` for the text
`main()` prints, `audit_outcome` for the free-text summary `main()` writes to the operator-action
log -- so this module never hand-writes a `print` call of its own per command.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from app.db.lifecycle import owner_engine
from app.gateway_provisioning import GatewayAdminClient
from app.migration_settings import get_migration_settings
from app.operator.audit import UNSCOPED_TENANT_ID, record_action
from app.operator.create import create_tenant
from app.operator.erase import erase_tenant, record_erasure
from app.operator.listing import TenantListing, list_tenants
from app.operator.suspend import set_tenant_suspended

OPERATOR_IDENTITY_ENV_VAR = "OPERATOR_IDENTITY"


class CommandResult(Protocol):
    """What every command's result type provides to `_run` below: text to print, and the
    free-text summary the audit log records. The two are not always the same string -- `create`
    and `erase` print more than their audit line says, and `list` prints something different
    from its audit line entirely -- so a result type provides both rather than `_run` trying to
    derive one from the other."""

    def render(self) -> str: ...

    @property
    def audit_outcome(self) -> str: ...


CommandFn = Callable[
    [AsyncConnection, argparse.Namespace, GatewayAdminClient | None],
    Awaitable[tuple[CommandResult, UUID | None]],
]


def _operator_identity() -> str:
    """Who is running this command, for the audit log. Never guessed from the database
    connection (that is always `app_owner`, the same for every operator) -- from the
    environment, defaulting to the OS user only as a local-development convenience."""
    return os.environ.get(OPERATOR_IDENTITY_ENV_VAR) or os.environ.get("USER", "unknown")


async def _dispatch_list(
    conn: AsyncConnection,
    args: argparse.Namespace,
    admin_client: GatewayAdminClient | None,
) -> tuple[TenantListing, None]:
    # `list` has no single target tenant -- it targets every tenant -- so its audit row uses the
    # documented sentinel (see app.operator.audit), never a real id.
    return TenantListing(tenants=await list_tenants(conn)), None


async def _dispatch_suspend_or_unsuspend(
    conn: AsyncConnection,
    args: argparse.Namespace,
    admin_client: GatewayAdminClient | None,
    *,
    suspended: bool,
) -> tuple[CommandResult, UUID]:
    result = await set_tenant_suspended(conn, args.identifier, suspended=suspended)
    return result, result.tenant_id


async def _dispatch_suspend(
    conn: AsyncConnection, args: argparse.Namespace, admin_client: GatewayAdminClient | None
) -> tuple[CommandResult, UUID]:
    return await _dispatch_suspend_or_unsuspend(conn, args, admin_client, suspended=True)


async def _dispatch_unsuspend(
    conn: AsyncConnection, args: argparse.Namespace, admin_client: GatewayAdminClient | None
) -> tuple[CommandResult, UUID]:
    return await _dispatch_suspend_or_unsuspend(conn, args, admin_client, suspended=False)


async def _dispatch_create(
    conn: AsyncConnection,
    args: argparse.Namespace,
    admin_client: GatewayAdminClient | None,
) -> tuple[CommandResult, UUID]:
    result = await create_tenant(
        conn,
        tenant_name=args.tenant_name,
        residency=args.residency,
        admin_email=args.admin_email,
        model=args.model,
        isolation_tier=args.isolation_tier,
        dedicated_db_admin_url=args.dedicated_db_admin_url,
        issuer=args.issuer,
        subject=args.subject,
        admin_client=admin_client,
    )
    return result, result.tenant_id


async def _dispatch_erase(
    conn: AsyncConnection,
    args: argparse.Namespace,
    admin_client: GatewayAdminClient | None,
) -> tuple[CommandResult, UUID]:
    result = await erase_tenant(
        conn,
        args.identifier,
        dry_run=args.dry_run,
        dedicated_db_admin_url=args.dedicated_db_admin_url,
        admin_client=admin_client,
    )
    if not result.dry_run:
        # Written whether or not every step succeeded (see app.operator.erase's module
        # docstring): a partial failure must still be visible on the erasure record, and a
        # re-run only needs to retry what this record shows as not yet done.
        await record_erasure(conn, result)
    return result, result.tenant_id


# Every command's dispatch function, keyed by its `build_parser()` subcommand name. Private:
# a test drives a command through `main()`, never this dict directly.
_COMMANDS: dict[str, CommandFn] = {
    "list": _dispatch_list,
    "suspend": _dispatch_suspend,
    "unsuspend": _dispatch_unsuspend,
    "create": _dispatch_create,
    "erase": _dispatch_erase,
}


async def _run(
    command: str,
    args: argparse.Namespace,
    engine: AsyncEngine,
    admin_client: GatewayAdminClient | None,
) -> int:
    started_at = datetime.now(UTC)
    outcome = "error"
    target_tenant_id: UUID | None = None
    error: str | None = None
    exit_code = 1
    try:
        async with engine.begin() as conn:
            result, target_tenant_id = await _COMMANDS[command](conn, args, admin_client)
        print(result.render())
        outcome = result.audit_outcome
        exit_code = 0
    except Exception as exc:  # noqa: BLE001 - recorded below, then re-raised as a nonzero exit
        # A lookup failure (TenantNotFoundError/AmbiguousTenantNameError) never resolved a
        # real tenant, so `target_tenant_id` stays None here -- the sentinel below is the
        # only honest target id for this row, same as any other failure.
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
        "--isolation-tier",
        dest="isolation_tier",
        default="pooled",
        choices=["pooled", "dedicated"],
        help="'pooled' (default) shares the deployment's one database; 'dedicated' provisions "
        "this tenant its own physical database (#71, ADR-0002) -- needs "
        "--dedicated-db-admin-url the first time",
    )
    create_parser.add_argument(
        "--dedicated-db-admin-url",
        dest="dedicated_db_admin_url",
        default=None,
        help="required only when --isolation-tier=dedicated and this tenant's database has not "
        "already been provisioned: an admin connection string (CREATEDB privilege) to the "
        "Postgres server that will host it. Used only for this invocation, never stored; "
        "redacted from the operator-action log.",
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

    erase_parser = sub.add_parser(
        "erase",
        help="Irreversibly erase a suspended tenant everywhere its data lives (ADR-0010). "
        "Refuses to run against a tenant that is not currently suspended.",
    )
    erase_parser.add_argument("identifier", help="tenant id or unambiguous name")
    erase_parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="report every step erase would take without removing anything",
    )
    erase_parser.add_argument(
        "--dedicated-db-admin-url",
        dest="dedicated_db_admin_url",
        default=None,
        help="required only to actually drop a dedicated tenant's database: an admin connection "
        "(DROP DATABASE privilege) to the Postgres server hosting it. Used only for this "
        "invocation, never stored; redacted from the operator-action log.",
    )
    return parser


async def _main_async(
    args: argparse.Namespace,
    *,
    engine: AsyncEngine | None,
    admin_client: GatewayAdminClient | None,
) -> int:
    if engine is not None:
        try:
            return await _run(args.command, args, engine, admin_client)
        finally:
            await engine.dispose()
    dsn = get_migration_settings().database_url_migrations.get_secret_value()
    async with owner_engine(dsn) as owned_engine:
        return await _run(args.command, args, owned_engine, admin_client)


def main(
    argv: list[str] | None = None,
    *,
    engine: AsyncEngine | None = None,
    admin_client: GatewayAdminClient | None = None,
) -> int:
    """The one public entry point (spec A5 / #115): parses `argv`, dispatches to the matching
    command, records the operator-action audit row, prints the result, and returns the process
    exit code. Dispatch (`_COMMANDS`, `_run`) stays private.

    With no `engine`, this builds one from the owner DSN
    (`app.migration_settings.get_migration_settings`) via `app.db.lifecycle.owner_engine`; either
    way -- built here or injected -- this call disposes it before returning (see the module
    docstring for why disposal never splits across two owners). `admin_client` is forwarded
    verbatim to whichever of `create`/`erase` the command names -- the only two that ever touch
    the gateway; `list`/`suspend`/`unsuspend` ignore it.

    Callable from a plain script (`scripts/operator.py`) or from inside an already-running event
    loop alike (an async test driving this function directly): with no loop running, this runs
    the command via a plain `asyncio.run()`; with one already running, `asyncio.run()` cannot be
    called here, so the whole call instead runs to completion on a fresh loop of its own, in one
    worker thread, and this blocks until it finishes.
    """
    args = build_parser().parse_args(argv)

    def _invoke() -> int:
        return asyncio.run(_main_async(args, engine=engine, admin_client=admin_client))

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _invoke()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_invoke).result()


if __name__ == "__main__":
    raise SystemExit(main())
