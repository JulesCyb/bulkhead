"""The conversation retention job (ADR-0006, Spec 4 / #35).

Deletes each tenant's expired conversations -- and, via the cascading foreign key (migration
0020), their messages -- one tenant at a time, entirely through `tenant_session(ctx)`: the exact
same tenant-bound, RLS-scoped `app`-role access path every other request in this project uses.
The per-tenant deletion never runs as a superuser, never as `app_owner`, and never against a
bypass-RLS policy.

Enumerating *which* tenants to visit is the one genuinely cross-tenant step this job needs, and it
reuses the operator tool's existing narrow escape hatch for exactly that
(`control.enumerate_tenants()`, migration 0012, `app/operator/listing.py`) rather than inventing a
second one: a `SECURITY DEFINER` function, owned by `app_owner`, gated on
`current_user = 'app_owner'` plus a flag it sets and restores itself. The connection passed to
`run_retention_job` is used for exactly that one read -- listing tenants -- and never for
anything touching `conversations`/`messages`; the moment a tenant is chosen, this job opens its
tenant's record read and its own `tenant_session(ctx)` (a distinct connection, routed by that
record the same way ADR-0002 routes any request) to delete its own expired rows.

Each tenant's cutoff is its own `settings["retention_days"]` (`app/tenant_settings.py`) if it has
set one, else the documented `DEFAULT_RETENTION_DAYS` -- read fresh, per tenant, from that
tenant's own record (`app.tenant_record.TenantRecord`, built by the very function a request's
context resolution uses: `ControlRepository.get_tenant_record`, #104/#105), the same settings
object `content_tracing_opt_in` and `model` are read from, so the setting is actually honored
rather than merely readable.

That raw stored value is never used as-is: it is passed through `app.tenant_settings.
effective_retention_days`, which clamps it to `Settings.max_retention_days` (#84, ADR-0006; GDPR
Art. 5(1)(e)) -- the read-side fail-safe for a row whose stored value predates a later, lower cap,
or was written before the cap existed at all (the write side, `app.operator.create`'s
`_validate_retention_days`, already refuses anything above the *current* cap, but cannot protect
a row written under a previous, higher one). A clamp is logged once, naming the tenant and the
stored value, so the sweep stays quiet for every tenant it does not have to correct.

A suspended tenant is skipped outright (#106, ADR-0010): its record is read (the same one read
every tenant gets), `record.suspended` is checked before anything else, and a suspended tenant
gets one log line and no `tenant_session()` -- never `TenantSuspendedError` raised mid-sweep, which
would otherwise abort the whole job's `for` loop at whichever tenant happened to be suspended.
The enumerate / read-record / skip-suspended / build-job-context loop itself lives in
`app/tenant_jobs.py` (`visit_active_tenants`, #82), shared with the pending-action sweep.
The reason is CONTEXT.md's definition of suspension -- a state in which "nothing is deleted"
(ADR-0010) -- and it is the one deliberate exception to CLAUDE.md rule 2's "with no exception"
for retention; `app/db/session.py`'s module docstring lists where suspension itself is refused.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncConnection

from app.config import Settings, get_settings
from app.db.session import tenant_session
from app.repositories.conversations import ConversationsRepository
from app.tenant_jobs import JOB_IDENTITY_ID, visit_active_tenants
from app.tenant_settings import effective_retention_days

log = logging.getLogger(__name__)

# Re-exported for existing importers; defined once in `app/tenant_jobs.py`.
# `ConversationsRepository.delete_expired` issues a DELETE, never an INSERT, so no `created_by`
# default is ever consulted for it.
__all__ = ["JOB_IDENTITY_ID", "RetentionOutcome", "run_retention_job"]


@dataclass(frozen=True, slots=True)
class RetentionOutcome:
    """One tenant's result from a single run of the job."""

    tenant_id: UUID
    tenant_name: str
    deleted: int


async def run_retention_job(
    conn: AsyncConnection, *, settings: Settings | None = None
) -> list[RetentionOutcome]:
    """Visits every tenant the control plane currently knows about (`conn`, an `app_owner`
    connection, used only to enumerate them) and deletes that tenant's conversations -- and their
    messages, via cascade -- whose `last_activity_at` is older than that tenant's own retention
    cutoff. A suspended tenant is skipped (one log line, no `tenant_session()` opened for it --
    module docstring) rather than counted as visited. Returns one `RetentionOutcome` per
    non-suspended tenant visited, in the order `list_tenants` returns them.

    `settings` (default: `get_settings()`) supplies `max_retention_days` (#84, ADR-0006) -- the
    read-side clamp `effective_retention_days` applies to every tenant's own stored value below.
    A test that needs a specific cap without touching process env/the cached `get_settings()`
    passes its own `Settings(...)` here, the same seam `app.operator.create.create_tenant` uses.
    """
    settings = settings or get_settings()
    outcomes: list[RetentionOutcome] = []
    async for visit in visit_active_tenants(conn, job="retention"):
        tenant, record, ctx = visit.tenant, visit.record, visit.ctx
        stored_retention_days = record.settings.retention_days
        retention_days = effective_retention_days(
            record.settings, max_days=settings.max_retention_days
        )
        exceeds_cap = (
            stored_retention_days is not None
            and stored_retention_days > settings.max_retention_days
        )
        if exceeds_cap:
            log.warning(
                "retention: tenant %s (%r) stored retention_days=%s exceeds "
                "MAX_RETENTION_DAYS=%s; clamping to %s",
                tenant.tenant_id,
                tenant.name,
                stored_retention_days,
                settings.max_retention_days,
                retention_days,
            )
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        async with tenant_session(ctx) as session:
            deleted = await ConversationsRepository().delete_expired(
                session, ctx, older_than=cutoff
            )
        outcomes.append(
            RetentionOutcome(tenant_id=tenant.tenant_id, tenant_name=tenant.name, deleted=deleted)
        )
    return outcomes
