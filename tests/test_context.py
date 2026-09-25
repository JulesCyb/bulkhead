import uuid

import pytest

from app.context import RequestContext


def test_context_is_immutable_and_checks_roles():
    ctx = RequestContext(
        tenant_id=uuid.uuid4(), identity_id=uuid.uuid4(), roles=frozenset({"admin"})
    )
    assert ctx.has_role("admin")
    ctx.require_role("admin")
    with pytest.raises(PermissionError):
        ctx.require_role("owner")
    with pytest.raises(AttributeError):  # frozen dataclass
        ctx.tenant_id = uuid.uuid4()  # type: ignore[misc]


def test_trace_attributes_contain_only_identifiers():
    ctx = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())
    attrs = ctx.trace_attributes()
    assert set(attrs) == {"tenant_id", "identity_id", "request_id"}
    assert attrs["tenant_id"] == str(ctx.tenant_id)
    assert attrs["identity_id"] == str(ctx.identity_id)
