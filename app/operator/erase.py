"""The `erase` command (Spec 9 / #72, ADR-0010, #114): the operator tool's one irreversible
command, removing a suspended tenant from every place its data lives and writing a permanent
record of exactly what was removed.

Reads the tenant's suspension state and isolation tier through
`app.repositories.control.ControlRepository.get_record` -- never SQL of its own against
`control.tenants` -- and refuses to run against one that is not currently suspended
(`TenantNotSuspendedError`), changing nothing: suspension precedes erasure by construction, not
by convention. Otherwise, per tenant, in order:

1. Revoke the tenant's gateway credential and delete its secret file
   (`app.gateway_provisioning.revoke_gateway_credential`, Spec 7 / #53, called with `conn=conn` so
   its own control-plane write joins this function's transaction) -- a no-op, not an error, if
   already revoked.
2. Request deletion of the tenant's traces by tenant id -- a per-tenant deletion capability Spec 8
   has not yet built a real backend for; `app.observability.delete_tenant_traces` is the stub this
   calls through by default (see that module's docstring), replaceable by a test's own fake via
   `trace_deleter=`.
3. For a dedicated tenant, drop its own physical database and both of its tenant-secret files
   (`app.operator.dedicated_db.drop_dedicated_database`, #71/#72) -- this removes every row that
   ever lived in that database, tenant-table registry included, along with the database itself.
4. Delete the tenant's own row from `public.tenants` on the pooled connection: every registered
   tenant table (`app.db.tenant_tables.TENANT_TABLES`) and `control.tenants` itself reference it
   `ON DELETE CASCADE` (migrations 0001, 0002, 0009, 0020, 0021, 0022, 0031, 0035), so one DELETE
   removes every pooled row that references it, and the control-plane record along with it.

Each step's outcome ("removed" | "already absent" | "requested" | "failed: <message>" | "skipped:
...") is accumulated into `EraseResult.steps` regardless of whether an earlier step raised -- a
step's own exception is caught and recorded, never left to abort the whole command -- and the
caller (`app.operator.cli`) writes it, together with the computed backup horizon, to
`control.tenant_erasures` whether or not every step succeeded, so a partial failure is visible.
Step 4 (the tenant row) only ever runs once every earlier step has actually succeeded or was
already a no-op; if any of them failed, it is recorded as "skipped" instead, deliberately leaving
the tenant resolvable so a re-run only retries what is not yet "removed"/"already
absent"/"requested" rather than a second, now-unresolvable, attempt.

Idempotent/re-runnable by construction: revocation, secret-file deletion, database drop, and the
final tenant-row delete are each individually a no-op against something already gone (`DROP
DATABASE IF EXISTS`, `Path.unlink(missing_ok=True)`, a DELETE matching zero rows), so re-running
`erase` after a partial failure repeats only the remaining work and never re-raises on what a
previous run already finished.

`--dry-run` (`dry_run=True`) reports what each step *would* do, from the tenant's current
control-plane state alone, without calling any of the four steps above -- no row, file, or
credential is touched.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.config import Settings, get_settings
from app.gateway_provisioning import GatewayAdminClient, revoke_gateway_credential
from app.observability import delete_tenant_traces
from app.operator.dedicated_db import drop_dedicated_database
from app.operator.lookup import resolve_tenant
from app.repositories.control import ControlRepository

TraceDeleter = Callable[[UUID], Awaitable[None]]

_FAILED_PREFIX = "failed: "


class TenantNotSuspendedError(RuntimeError):
    """`erase` refuses to run against a tenant that is not currently suspended -- suspension is
    the required, reversible first step (ADR-0010); nothing is changed when this is raised."""


@dataclass(frozen=True, slots=True)
class ErasureStepResult:
    step: str
    outcome: str


@dataclass(frozen=True, slots=True)
class EraseResult:
    tenant_id: UUID
    name: str
    isolation_tier: str
    dry_run: bool
    steps: tuple[ErasureStepResult, ...]
    backup_horizon: datetime | None  # None only for a dry run -- no erasure record is written

    @property
    def any_step_failed(self) -> bool:
        return any(step.outcome.startswith(_FAILED_PREFIX) for step in self.steps)

    def render(self) -> str:
        """Byte-identical to what `app.operator.cli`'s retired `_run_erase` printed (spec A5 /
        #115): one line per step, then -- for a real (non-dry-run) erasure -- the backup horizon.
        Never the `audit_outcome` line below; that one was never printed by `_run_erase` either,
        only written to the operator-action log."""
        lines = [f"{step.step}: {step.outcome}" for step in self.steps]
        if not self.dry_run:
            assert self.backup_horizon is not None
            lines.append(f"backup horizon: {self.backup_horizon.isoformat()}")
        return "\n".join(lines)

    @property
    def audit_outcome(self) -> str:
        """The one line `app.operator.cli` writes to the operator-action log for this command."""
        if self.dry_run:
            return f"dry-run: {self.name!r} ({self.tenant_id}) -- no changes made"
        prefix = "partial" if self.any_step_failed else "ok"
        verb = "partially erased (see steps above)" if self.any_step_failed else "erased"
        return f"{prefix}: {verb} {self.name!r} ({self.tenant_id})"

    def as_details(self) -> dict[str, object]:
        """The JSON-shaped payload `app.operator.cli` writes into
        `control.tenant_erasures.details` -- never called for a dry run."""
        return {
            "isolation_tier": self.isolation_tier,
            "steps": {step.step: step.outcome for step in self.steps},
            "backup_horizon": self.backup_horizon.isoformat() if self.backup_horizon else None,
        }


def compute_backup_horizon(*, settings: Settings, now: datetime | None = None) -> datetime:
    """The date by which even the last backup copy of an erased tenant's data will have aged out
    -- the true erasure deadline (ADR-0010), computed from `settings.backup_retention_days`."""
    now = now if now is not None else datetime.now(UTC)
    return now + timedelta(days=settings.backup_retention_days)


async def erase_tenant(
    conn: AsyncConnection,
    identifier: str,
    *,
    dry_run: bool = False,
    dedicated_db_admin_url: str | None = None,
    settings: Settings | None = None,
    admin_client: GatewayAdminClient | None = None,
    trace_deleter: TraceDeleter | None = None,
) -> EraseResult:
    """Erase (or dry-run erase) the tenant named by `identifier` (id or unambiguous name). See
    module docstring for the full contract.

    `admin_client` is forwarded verbatim to `app.gateway_provisioning.revoke_gateway_credential`
    as its own test seam; that call also gets this function's own `conn` (#114), so its
    control-plane write shares this transaction rather than opening a second one. `trace_deleter`
    replaces `app.observability.delete_tenant_traces` -- the test seam for Spec 8's not-yet-built
    per-tenant trace deletion. `dedicated_db_admin_url` is only consulted, and only required, when
    this tenant is on the dedicated tier and its database has not already been dropped (see
    `app.operator.dedicated_db.drop_dedicated_database`).
    """
    ref = await resolve_tenant(conn, identifier)
    settings = settings or get_settings()
    trace_deleter = trace_deleter or delete_tenant_traces

    record = await ControlRepository().get_record(conn, ref.tenant_id)
    if not record.suspended:
        raise TenantNotSuspendedError(
            f"tenant {ref.name!r} ({ref.tenant_id}) is not suspended; erase refuses to run "
            "against an active tenant -- suspend it first (see app.operator.suspend)."
        )

    isolation_tier: str = record.isolation_tier
    database_alias: str | None = record.database_alias

    if dry_run:
        steps = [
            ErasureStepResult(
                "gateway_credential",
                "would revoke the gateway credential and delete its secret file",
            ),
            ErasureStepResult("traces", "would request per-tenant trace deletion"),
        ]
        if isolation_tier == "dedicated":
            steps.append(
                ErasureStepResult(
                    "dedicated_database",
                    f"would drop database {database_alias!r} and its secret files",
                )
            )
        steps.append(
            ErasureStepResult(
                "tenant_row",
                "would delete the tenant row (cascades to every registered tenant table and "
                "the control-plane record)",
            )
        )
        return EraseResult(
            tenant_id=ref.tenant_id,
            name=ref.name,
            isolation_tier=isolation_tier,
            dry_run=True,
            steps=tuple(steps),
            backup_horizon=None,
        )

    steps = []

    try:
        revoked = await revoke_gateway_credential(
            ref.tenant_id,
            settings=settings,
            admin_client=admin_client,
            conn=conn,
        )
        steps.append(
            ErasureStepResult("gateway_credential", "removed" if revoked else "already absent")
        )
    except Exception as exc:  # noqa: BLE001 - recorded, not re-raised: see module docstring
        steps.append(ErasureStepResult("gateway_credential", f"{_FAILED_PREFIX}{exc}"))

    try:
        await trace_deleter(ref.tenant_id)
        steps.append(ErasureStepResult("traces", "requested"))
    except Exception as exc:  # noqa: BLE001
        steps.append(ErasureStepResult("traces", f"{_FAILED_PREFIX}{exc}"))

    if isolation_tier == "dedicated":
        assert database_alias is not None  # enforced by migration 0005's CHECK constraint
        try:
            outcome = await drop_dedicated_database(
                alias=database_alias, admin_url=dedicated_db_admin_url
            )
            steps.append(ErasureStepResult("dedicated_database", outcome))
        except Exception as exc:  # noqa: BLE001
            steps.append(ErasureStepResult("dedicated_database", f"{_FAILED_PREFIX}{exc}"))

    # The tenant row -- and, via cascade, every pooled row that references it -- is only ever
    # deleted once every step above has actually succeeded (or was already a no-op). If any of
    # them failed, the tenant stays resolvable by `resolve_tenant` for a re-run: skipping this
    # step here, rather than deleting the row regardless, is what makes "re-run erase after a
    # partial failure" a real retry instead of a second, now-unresolvable, erase attempt.
    if any(step.outcome.startswith(_FAILED_PREFIX) for step in steps):
        steps.append(
            ErasureStepResult(
                "tenant_row", "skipped: an earlier step failed -- re-run erase once it is fixed"
            )
        )
    else:
        # A SAVEPOINT, not the outer transaction `conn` runs in (opened by
        # app.operator.cli._run): if this DELETE itself fails, only this nested transaction
        # rolls back, so the outer transaction is still healthy afterwards for app.operator.cli
        # to write the erasure record and the operator-action-log entry on this same connection.
        try:
            async with conn.begin_nested():
                result = await conn.execute(
                    text("DELETE FROM tenants WHERE id = :tid"), {"tid": ref.tenant_id}
                )
            steps.append(
                ErasureStepResult("tenant_row", "removed" if result.rowcount else "already absent")
            )
        except Exception as exc:  # noqa: BLE001
            steps.append(ErasureStepResult("tenant_row", f"{_FAILED_PREFIX}{exc}"))

    return EraseResult(
        tenant_id=ref.tenant_id,
        name=ref.name,
        isolation_tier=isolation_tier,
        dry_run=False,
        steps=tuple(steps),
        backup_horizon=compute_backup_horizon(settings=settings),
    )


async def record_erasure(conn: AsyncConnection, result: EraseResult) -> None:
    """Write one row to `control.tenant_erasures` (migration 0004, #114) for a completed
    (successful or partial) `erase_tenant` run -- never called for a dry run. Delegates to
    `ControlRepository.record_erasure`, which takes `tenant_id` as a plain value, not a foreign
    key: the row must document and outlive the tenant row `erase_tenant` may have just deleted,
    exactly as migration 0004 requires."""
    await ControlRepository().record_erasure(
        conn,
        tenant_id=result.tenant_id,
        details_json=json.dumps(result.as_details(), default=str),
    )
