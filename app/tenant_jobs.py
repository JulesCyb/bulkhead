"""The one loop every scheduled per-tenant job shares (#82): visit each non-suspended tenant with
a job context, one tenant at a time.

Two jobs use it today -- the conversation retention job (`app/retention.py`, ADR-0006) and the
pending-action sweep (`app/pending_action_sweep.py`, ADR-0007) -- and both need exactly the same
four steps before doing their own work, so the steps live here once instead of being copied:

1. **Enumerate** every tenant the control plane knows about on the owner connection the caller
   passes in (`control.enumerate_tenants()`, migration 0012, via `app/operator/listing.py`) --
   the one genuinely cross-tenant read a job needs, reusing the operator tool's narrow
   `SECURITY DEFINER` escape hatch rather than inventing a second one. That connection is used
   for nothing else.
2. **Build the tenant record** (`app.tenant_record.TenantRecord`) with the very function a
   request's context resolution uses (`ControlRepository.get_tenant_record`, #104/#105), read
   fresh per tenant, never cached across runs.
3. **Skip a suspended tenant** with one log line and no `tenant_session()` opened for it (#106,
   ADR-0010: suspension is a state in which nothing is deleted or changed) -- never
   `TenantSuspendedError` raised mid-loop, which would abort the whole job at whichever tenant
   happened to be suspended.
4. **Yield a job context** -- the tenant's id, `JOB_IDENTITY_ID`, and the record just read --
   that the job passes to `tenant_session(ctx)`, the same tenant-bound, RLS-scoped `app`-role
   path every request uses. No job ever runs as a superuser, as `app_owner`, or against a
   bypass-RLS policy for its per-tenant work.

The job context has no delegation means (`ctx.means is None`, ADR-0005): a scheduled job acts on
nobody's behalf, so an audit row it writes carries null means columns -- the correct value, not a
gap (`app/repositories/approval_audit.py`).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncConnection

from app.context import RequestContext
from app.db.session import tenant_record_session
from app.operator.listing import list_tenants
from app.repositories.control import ControlRepository, TenantSummary
from app.tenant_record import TenantRecord

log = logging.getLogger(__name__)

# A job has no acting person or agent identity behind it -- it is a scheduled sweep, not a request
# on anyone's behalf. It is only ever used as the per-transaction `app.identity_id` setting; no job
# may INSERT into a table whose `created_by` default (a NOT NULL foreign key to
# `control.identities`) would consult it. Same nil-UUID-as-sentinel convention as
# `app/operator/audit.py`'s `UNSCOPED_TENANT_ID`.
JOB_IDENTITY_ID = UUID(int=0)


@dataclass(frozen=True, slots=True)
class TenantJobVisit:
    """One non-suspended tenant a job is about to work on: its enumeration facts (`tenant`, for
    naming it in outcomes and logs), its freshly read tenant record (`record`, the same object
    `ctx.tenant_record` carries -- here for a job that reads a tenant setting from it), and the
    job context to open `tenant_session(ctx)` with."""

    tenant: TenantSummary
    record: TenantRecord
    ctx: RequestContext


async def visit_active_tenants(conn: AsyncConnection, *, job: str) -> AsyncIterator[TenantJobVisit]:
    """Yields one `TenantJobVisit` per non-suspended tenant, in the order `list_tenants` returns
    them (module docstring, steps 1-4). `conn` is an `app_owner` connection used only to enumerate
    tenants. `job` prefixes the one log line a skipped suspended tenant gets (e.g. `"retention"`),
    so each job's log still says which job skipped it."""
    for tenant in await list_tenants(conn):
        async with tenant_record_session(tenant.tenant_id) as record_session:
            record = await ControlRepository().get_tenant_record(
                record_session, tenant_id=tenant.tenant_id
            )
        if record.suspended:
            log.info("%s: skipping suspended tenant %s (%r)", job, tenant.tenant_id, tenant.name)
            continue
        yield TenantJobVisit(
            tenant=tenant,
            record=record,
            ctx=RequestContext(
                tenant_id=tenant.tenant_id, identity_id=JOB_IDENTITY_ID, tenant_record=record
            ),
        )


__all__ = ["JOB_IDENTITY_ID", "TenantJobVisit", "visit_active_tenants"]
