"""The tenant record: what the control plane says about one tenant, read once per request
(#104, spec #92 "A2", ADR-0002, ADR-0008, ADR-0010, ADR-0011; `CONTEXT.md` "Tenant record").

One immutable value carrying the operator-owned facts a request needs about its tenant --
isolation tier, database alias, residency, suspension, gateway credential alias -- together with
the tenant's own validated, tenant-editable settings. `app.context_resolution` reads it through
`app.token_verifier.ControlPlaneReads.get_tenant_record` right after the membership check,
refuses a suspended one there, and attaches it to the `RequestContext` it returns
(`RequestContext.tenant_record`); `app.db.session.tenant_session` then routes by it instead of
reading the control plane again. It is never cached across requests: the next request reads it
afresh, so an operator's suspend or re-route takes effect on that very next request.

Why its own module rather than `app/context.py`: `app/context.py` is on `CLAUDE.md`'s "do not
touch without checking first" list, and the record is a control-plane value, not part of "who is
acting". Keeping it here means the context module gains exactly one optional field and one
import, and a later change to what the record carries (#105, #113) touches this file only.

The record deliberately makes no fail-closed decision of its own: a `residency` of `None` is
carried as `None`, and it is the resolvers (model, embeddings, tracing -- #105) that decide what
an unresolved residency means for their path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from uuid import UUID

from app.tenant_settings import TenantSettings

IsolationTier = Literal["pooled", "dedicated"]
"""The two isolation tiers (ADR-0002; the `control.tenants.isolation_tier` CHECK, migration
0005)."""


@dataclass(frozen=True, slots=True)
class TenantRecord:
    """The control-plane facts and tenant-editable settings of one tenant, as read at one moment.

    `database_alias` is `None` for a pooled tenant (the pooled database needs no alias).
    `tenant_id` is carried so a consumer can refuse a record that belongs to another tenant than
    the context it rides on (`tenant_session` does)."""

    tenant_id: UUID
    isolation_tier: IsolationTier = "pooled"
    database_alias: str | None = None
    residency: str | None = None
    suspended_at: datetime | None = None
    gateway_credential_alias: str | None = None
    settings: TenantSettings = field(default_factory=TenantSettings)

    @property
    def suspended(self) -> bool:
        return self.suspended_at is not None

    @classmethod
    def pooled_default(cls, tenant_id: UUID) -> TenantRecord:
        """The record of a tenant the control plane has no row for: pooled, not suspended, no
        residency, no gateway credential, default settings (ADR-0002's pooled default -- the
        same thing `tenant_session`'s own routing read concludes from a missing row)."""
        return cls(tenant_id=tenant_id)


__all__ = ["IsolationTier", "TenantRecord"]
