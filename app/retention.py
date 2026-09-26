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
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncConnection

from app.context import RequestContext
from app.db.session import tenant_record_session, tenant_session
from app.operator.listing import list_tenants
from app.repositories.control import ControlRepository
from app.repositories.conversations import ConversationsRepository
from app.tenant_settings import DEFAULT_RETENTION_DAYS

# This job has no acting person or agent identity behind it -- it is a scheduled sweep, not a
# request on anyone's behalf. It is only ever used as the per-transaction `app.identity_id`
# setting; `ConversationsRepository.delete_expired` issues a DELETE, never an INSERT, so no
# `created_by` default (a NOT NULL foreign key to `control.identities`) is ever consulted for it.
# Same nil-UUID-as-sentinel convention as `app/operator/audit.py`'s `UNSCOPED_TENANT_ID`.
JOB_IDENTITY_ID = UUID(int=0)


@dataclass(frozen=True, slots=True)
class RetentionOutcome:
    """One tenant's result from a single run of the job."""

    tenant_id: UUID
    tenant_name: str
    deleted: int


async def run_retention_job(conn: AsyncConnection) -> list[RetentionOutcome]:
    """Visits every tenant the control plane currently knows about (`conn`, an `app_owner`
    connection, used only to enumerate them) and deletes that tenant's conversations -- and their
    messages, via cascade -- whose `last_activity_at` is older than that tenant's own retention
    cutoff. Returns one `RetentionOutcome` per tenant visited, in the order `list_tenants` returns
    them.
    """
    outcomes: list[RetentionOutcome] = []
    for tenant in await list_tenants(conn):
        async with tenant_record_session(tenant.tenant_id) as record_session:
            record = await ControlRepository().get_tenant_record(
                record_session, tenant_id=tenant.tenant_id
            )
        ctx = RequestContext(
            tenant_id=tenant.tenant_id, identity_id=JOB_IDENTITY_ID, tenant_record=record
        )
        retention_days = record.settings.retention_days or DEFAULT_RETENTION_DAYS
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        async with tenant_session(ctx) as session:
            deleted = await ConversationsRepository().delete_expired(
                session, ctx, older_than=cutoff
            )
        outcomes.append(
            RetentionOutcome(tenant_id=tenant.tenant_id, tenant_name=tenant.name, deleted=deleted)
        )
    return outcomes
