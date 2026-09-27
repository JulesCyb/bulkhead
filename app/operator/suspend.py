"""The `suspend`/`unsuspend` commands (Spec 9 / #69, ADR-0010, #114): resolve a tenant by id or
unambiguous name (`app.operator.lookup`), then flip its suspension state through
`app.repositories.control.ControlRepository.set_suspended` -- which itself calls
`control.set_tenant_suspended()` (migration 0024), the one `SECURITY DEFINER` write path the
operator role is granted onto `control.tenants`.

Idempotent by construction: re-running `suspend` against an already-suspended tenant, or
`unsuspend` against an already-active one, is reported as a no-op (`changed=False`), never an
error -- the caller (`app.operator.cli`) turns that into the outcome string the audit log records.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncConnection

from app.operator.lookup import resolve_tenant
from app.repositories.control import ControlRepository


@dataclass(frozen=True, slots=True)
class SuspendResult:
    tenant_id: UUID
    name: str
    suspended: bool
    changed: bool
    suspended_at: datetime | None

    @property
    def audit_outcome(self) -> str:
        """The one line `app.operator.cli` both prints and writes to the operator-action log for
        this command (spec A5 / #115) -- derivable from `suspended`/`changed` alone: `suspend`
        always requests `suspended=True` and `unsuspend` always requests `suspended=False`, and
        this result is idempotent on that request, so `self.suspended` already says which verb
        applies without the caller passing it back in."""
        verb = "suspended" if self.suspended else "unsuspended"
        if not self.changed:
            return f"no-op: {self.name!r} ({self.tenant_id}) was already {verb}"
        return f"ok: {self.name!r} ({self.tenant_id}) is now {verb}"

    def render(self) -> str:
        """Byte-identical to what `app.operator.cli`'s retired `_run_suspend_or_unsuspend`
        printed: this one line, and only this line."""
        return self.audit_outcome


async def set_tenant_suspended(
    conn: AsyncConnection, identifier: str, *, suspended: bool
) -> SuspendResult:
    """Resolves `identifier` (id or unambiguous name) and sets its suspension state to
    `suspended`. Raises `app.operator.lookup.TenantNotFoundError`/`AmbiguousTenantNameError` if
    `identifier` does not resolve to exactly one tenant -- the same lookup every other command
    uses, never a second, looser resolution just for this one.
    """
    ref = await resolve_tenant(conn, identifier)
    outcome = await ControlRepository().set_suspended(conn, ref.tenant_id, suspended)
    return SuspendResult(
        tenant_id=ref.tenant_id,
        name=ref.name,
        suspended=outcome.suspended,
        changed=outcome.changed,
        suspended_at=outcome.suspended_at,
    )
