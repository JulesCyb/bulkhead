"""The context object: who is acting, for which tenant.

Built once per HTTP request (app/deps.py) and passed through agent run, tools, and
repositories. Nothing reads tenant or identity from global state.
"""

from __future__ import annotations

import dataclasses
import uuid
from dataclasses import dataclass, field
from typing import Literal, get_args
from uuid import UUID

Role = Literal["admin", "member", "support", "agent"]
"""The four legal membership roles (CONTEXT.md's "Role" glossary entry, ADR-0004, ADR-0005).

This is the single source of truth both `RequestContext.require_role` below and the database
check against: the `memberships_role_valid` CHECK constraint on `memberships.role`
(`migrations/versions/0009_tenant_memberships.py`) must name the exact same four strings.
`tests/test_rls_integration.py::test_role_type_matches_the_database_check_constraint` keeps the
two from silently drifting apart -- change one, change the other, or that test fails.
"""

ROLES: frozenset[str] = frozenset(get_args(Role))


MeansKind = Literal["agent", "credential"]
"""What carried a request's action out (ADR-0005, CONTEXT.md's "Delegation"/"Agent identity"
glossary entries): `"agent"` for delegation (an agent acting with a person's membership -- the
means is which agent handled it), `"credential"` for autonomous use (an agent identity acting on
its own -- the means is which credential authenticated it)."""


@dataclass(frozen=True, slots=True)
class Means:
    """Which agent handled a delegated request, or which credential authenticated an autonomous
    one -- attached to a `RequestContext` via `RequestContext.acting_through(...)`, never
    constructed directly by callers that only ever set the actor."""

    kind: MeansKind
    id: str


@dataclass(frozen=True, slots=True)
class RequestContext:
    tenant_id: UUID
    identity_id: UUID
    roles: frozenset[str] = field(default_factory=frozenset)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    means: Means | None = None
    """Set only via `acting_through(...)`. `None` for a context built the ordinary way (no
    means to report) -- the common, unchanged shape every existing caller still constructs."""

    def has_role(self, role: Role) -> bool:
        return role in self.roles

    def require_role(self, role: Role) -> None:
        if not self.has_role(role):
            raise PermissionError(f"role {role!r} required")

    def acting_through(self, kind: MeansKind, id: str) -> RequestContext:
        """Returns a new context with the same actor (tenant/identity/roles/request_id)
        and `means` set -- the object itself stays immutable and frozen."""
        return dataclasses.replace(self, means=Means(kind=kind, id=id))

    def trace_attributes(self) -> dict[str, str]:
        """Attributes for tracing/Langfuse — never content, identifiers only. Omits the means
        fields entirely when no means is set, so a plain, delegation-free context's trace stays
        exactly as it is today."""
        attrs = {
            "tenant_id": str(self.tenant_id),
            "identity_id": str(self.identity_id),
            "request_id": self.request_id,
        }
        if self.means is not None:
            attrs["means_kind"] = self.means.kind
            attrs["means_id"] = self.means.id
        return attrs
