"""One engine-lifecycle helper (spec A5 / #113, spec #95's "one engine-lifecycle helper"
decision): build an `AsyncEngine` from a DSN, yield it to the caller, dispose it once the caller's
block exits -- normally or by exception.

Before this module, that same three-line shape ("build, use, dispose in a `finally`") was
hand-written seven times: `app/operator/cli.py`'s `_run`, `scripts/retention.py`,
`scripts/migrate.py`'s alias enumeration, `app/gateway_provisioning.py`'s `_build_owner_engine`
(and its two callers' own `finally: await owner_engine.dispose()`), and `app/operator/
dedicated_db.py`'s three admin/owner engines.

This ticket (#113) wires it into the first three call sites above -- the operator CLI's own
`_run`, the retention entry point, and the migration runner's alias enumeration.
`app/gateway_provisioning.py` and `app/operator/dedicated_db.py` keep their own hand-written
copies until #114 rewires the operator commands onto this helper and the control repository
together; see that ticket.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


@asynccontextmanager
async def owner_engine(dsn: str, **engine_kwargs: object) -> AsyncIterator[AsyncEngine]:
    """Build an `AsyncEngine` from `dsn`, yield it to the caller, and dispose it once the
    caller's block exits -- normally or by exception. `**engine_kwargs` forwards to
    `create_async_engine` (e.g. `isolation_level="AUTOCOMMIT"`, needed by a `CREATE DATABASE`/
    `DROP DATABASE` caller -- no caller in this ticket needs it, but the parameter exists so #114
    can reuse this helper for `app/operator/dedicated_db.py` too instead of a second one)."""
    engine = create_async_engine(dsn, **engine_kwargs)
    try:
        yield engine
    finally:
        await engine.dispose()


__all__ = ["owner_engine"]
