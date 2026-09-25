"""The context object: who is acting, for which tenant.

Built once per HTTP request (app/deps.py) and passed through agent run, tools, and
repositories. Nothing reads tenant or identity from global state.
"""

from __future__ import annotations

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


@dataclass(frozen=True, slots=True)
class RequestContext:
    tenant_id: UUID
    identity_id: UUID
    roles: frozenset[str] = field(default_factory=frozenset)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def has_role(self, role: Role) -> bool:
        return role in self.roles

    def require_role(self, role: Role) -> None:
        if not self.has_role(role):
            raise PermissionError(f"role {role!r} required")

    def trace_attributes(self) -> dict[str, str]:
        """Attributes for tracing/Langfuse — never content, identifiers only."""
        return {
            "tenant_id": str(self.tenant_id),
            "identity_id": str(self.identity_id),
            "request_id": self.request_id,
        }
