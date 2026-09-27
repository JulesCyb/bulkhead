"""Runs the pending-action sweep once (ADR-0007, #82): marks every tenant's unanswered, overdue
pending actions `expired` and writes one `expired` audit event for each, through the same
tenant-bound, RLS-scoped access path every other request in this project uses. See
`app/pending_action_sweep.py` for the full contract.

Connects as the owner role (`DATABASE_URL_MIGRATIONS`, the same DSN `scripts/retention.py`,
`scripts/migrate.py` and `scripts/operator.py` use) only to enumerate tenants
(`control.enumerate_tenants()`). The per-tenant work never uses this connection: it always goes
through the ordinary `app`-role `tenant_session()`. Suspended tenants are skipped.

Pending actions expire after minutes (`PENDING_ACTION_EXPIRY_SECONDS`), so schedule this far more
often than the daily retention job -- every few minutes, via cron, a Kubernetes CronJob, or a
one-shot Compose service; there is no scheduler built into this script itself. Running it more
often than needed is harmless: a second run finds nothing to do.

    uv run python scripts/sweep_pending_actions.py
"""

from __future__ import annotations

import asyncio

from app.db.lifecycle import owner_engine
from app.migration_settings import get_migration_settings
from app.pending_action_sweep import run_pending_action_sweep


async def _main() -> None:
    dsn = get_migration_settings().database_url_migrations.get_secret_value()
    async with owner_engine(dsn) as engine:
        async with engine.begin() as conn:
            outcomes = await run_pending_action_sweep(conn)

    for outcome in outcomes:
        if outcome.expired:
            print(
                f"sweep: tenant {outcome.tenant_id} ({outcome.tenant_name!r}): "
                f"expired {len(outcome.expired)} pending action(s)"
            )
    total = sum(len(outcome.expired) for outcome in outcomes)
    print(f"sweep: visited {len(outcomes)} tenant(s), expired {total} pending action(s)")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
