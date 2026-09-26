"""Runs the conversation retention job once (ADR-0006, Spec 4 / #35): deletes every tenant's
expired conversations -- and their messages, via cascade -- through the same tenant-bound,
RLS-scoped access path every other request in this project uses. See `app/retention.py` for the
full contract.

Connects as the owner role (`DATABASE_URL_MIGRATIONS`, the same DSN `scripts/migrate.py` and
`scripts/operator.py` use) only to enumerate tenants (`control.enumerate_tenants()`, the operator
tool's own narrow cross-tenant read, `app/operator/listing.py`). The actual per-tenant deletion
never uses this connection: it always goes through the ordinary `app`-role `tenant_session()`.

Run it however this deployment schedules recurring jobs -- cron, a Kubernetes CronJob, a
one-shot Compose service -- there is no scheduler built into this script itself:

    uv run python scripts/retention.py
"""

from __future__ import annotations

import asyncio

from app.db.lifecycle import owner_engine
from app.migration_settings import get_migration_settings
from app.retention import run_retention_job


async def _main() -> None:
    dsn = get_migration_settings().database_url_migrations.get_secret_value()
    async with owner_engine(dsn) as engine:
        async with engine.begin() as conn:
            outcomes = await run_retention_job(conn)

    for outcome in outcomes:
        if outcome.deleted:
            print(
                f"retention: tenant {outcome.tenant_id} ({outcome.tenant_name!r}): "
                f"deleted {outcome.deleted} expired conversation(s)"
            )
    total = sum(outcome.deleted for outcome in outcomes)
    print(f"retention: swept {len(outcomes)} tenant(s), deleted {total} expired conversation(s)")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
