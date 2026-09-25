"""The operator-action audit log (Spec 9 / #68, ADR-0010): every invocation of the operator
tool -- successful, failed, or a no-op -- is written to `control.operator_actions`
(migration 0004), append-only by grant: `app_owner` (the role every operator-tool connection
uses) holds `INSERT` only on that table, no `SELECT`/`UPDATE`/`DELETE`, so this module can record
an action but never amend or read one back. See CLAUDE.md's "Do not touch" list.

`record_action` is called exactly once per command invocation, in a transaction separate from
the command's own work (see `app.operator.cli`): if the command's own transaction rolls back on
failure, the audit record describing that failure must still commit, so it cannot share that
transaction.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

# `control.operator_actions.tenant_id` is NOT NULL (migration 0004): every row names a target
# tenant. A command like `list` targets every tenant, not one -- there is no real tenant id to
# put there. The nil UUID is the documented sentinel for "this action was not scoped to a single
# tenant", chosen over allowing NULL so the append-only table's schema never has to change for a
# distinction the operator tool, not the database, is responsible for explaining.
UNSCOPED_TENANT_ID = UUID(int=0)

# Argument keys never written to the log, even under an operator's own audit trail -- secrets
# never touch a database row (ADR-0011), only files.
_REDACTED_ARG_KEYS = {"password", "secret", "token", "credential", "dsn", "database_url"}


def redact_args(args: dict[str, Any]) -> dict[str, Any]:
    """A copy of `args` with any secret-shaped key's value replaced -- never the key itself, so
    the log still shows which argument was supplied."""
    return {
        key: ("<redacted>" if key.lower() in _REDACTED_ARG_KEYS else value)
        for key, value in args.items()
    }


async def record_action(
    conn: AsyncConnection,
    *,
    operator: str,
    command: str,
    target_tenant_id: UUID | None,
    args: dict[str, Any],
    outcome: str,
    started_at: datetime,
    finished_at: datetime,
    error: str | None = None,
) -> None:
    """Writes one row to `control.operator_actions`. `outcome` is a short, free-text status
    ("ok", "no-op", "error", ...) -- the command decides its own vocabulary; this module only
    stores it.
    """
    details: dict[str, Any] = {
        "operator": operator,
        "args": redact_args(args),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_ms": round((finished_at - started_at).total_seconds() * 1000),
        "outcome": outcome,
    }
    if error is not None:
        details["error"] = error

    await conn.execute(
        text(
            "INSERT INTO control.operator_actions (tenant_id, action, details) "
            "VALUES (:tenant_id, :action, CAST(:details AS jsonb))"
        ),
        {
            "tenant_id": str(target_tenant_id)
            if target_tenant_id is not None
            else str(UNSCOPED_TENANT_ID),
            "action": command,
            "details": json.dumps(details, default=str),
        },
    )
