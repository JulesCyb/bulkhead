"""The read-only tenant listing (Spec 9 / #68, #114): the operator tool's first real command. Lets
an operator or auditor see every tenant's isolation tier, residency, database alias, and
suspension state without a direct database query.

Reads through `app.repositories.control.ControlRepository.enumerate_tenants`, which calls
`control.enumerate_tenants()` (migration 0012/0024): a narrow, `current_user`-gated, `SECURITY
DEFINER` cross-tenant read -- never a plain role-scoped bypass policy on `control.tenants` or
`public.tenants` directly, which (see that migration's docstring) would leak through
`control.tenants_view` to the `app` role. Not granted to `app`; only `app_owner` may call it.
`app.operator.lookup.resolve_tenant` reads the exact same enumeration to resolve one tenant by id
or name, rather than this module and that one issuing two different queries against the same
function.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncConnection

from app.repositories.control import ControlRepository, TenantSummary

__all__ = ["TenantListing", "TenantSummary", "list_tenants"]


async def list_tenants(conn: AsyncConnection) -> list[TenantSummary]:
    """Every tenant in the control plane, with its lifecycle facts."""
    return await ControlRepository().enumerate_tenants(conn)


@dataclass(frozen=True, slots=True)
class TenantListing:
    """The `list` command's own result type (spec A5 / #115): wraps `list_tenants`'s plain
    `list[TenantSummary]` so `app.operator.cli` has one result-protocol object (`render()`,
    `audit_outcome`) per command, matching `SuspendResult`/`CreateTenantResult`/`EraseResult`.
    `list_tenants` itself keeps returning the plain list -- every other caller of it (the
    tenant-lookup helper, the tests that inspect an individual `TenantSummary`) has no use for
    this wrapper."""

    tenants: list[TenantSummary]

    def render(self) -> str:
        """Byte-identical to what `app.operator.cli`'s retired `_run_list` printed: either the
        one "no tenants" line, or one line per tenant -- never both."""
        if not self.tenants:
            return "No tenants in the control plane."
        return "\n".join(
            f"{t.tenant_id}  {t.name!r:30}  tier={t.isolation_tier:9}  "
            f"residency={t.residency or '-':6}  alias={t.database_alias or '-':12}  "
            f"suspended={t.suspended}"
            for t in self.tenants
        )

    @property
    def audit_outcome(self) -> str:
        """The one line `app.operator.cli` writes to the operator-action log for this command --
        `list` has no single target tenant, so this is the only summary it ever records."""
        return f"ok: listed {len(self.tenants)} tenant(s)"
