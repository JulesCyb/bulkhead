import uuid

import pytest

from app.context import ROLES, Means, RequestContext


def test_roles_are_the_four_legal_values():
    """Acceptance (#28): the Python single source of truth names exactly the four roles
    CONTEXT.md documents, string for string."""
    assert ROLES == {"admin", "member", "support", "agent"}


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


def test_plain_context_has_no_means_and_omits_it_from_trace_attributes():
    """Acceptance #43: a context built the ordinary way still exposes no means and its trace
    attributes omit any means fields."""
    ctx = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())
    assert ctx.means is None
    attrs = ctx.trace_attributes()
    assert "means_kind" not in attrs
    assert "means_id" not in attrs


def test_acting_through_sets_means_and_keeps_actor_and_immutability():
    """Acceptance #43: `acting_through(...)` returns a context with the original actor fields
    untouched and the means set, and the object stays otherwise immutable."""
    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    ctx = RequestContext(tenant_id=tenant_id, identity_id=identity_id, roles=frozenset({"member"}))

    delegated = ctx.acting_through("agent", "assistant")

    assert delegated is not ctx
    assert ctx.means is None  # the original context is untouched
    assert delegated.means == Means(kind="agent", id="assistant")
    assert delegated.tenant_id == tenant_id
    assert delegated.identity_id == identity_id
    assert delegated.roles == frozenset({"member"})
    assert delegated.request_id == ctx.request_id
    with pytest.raises(AttributeError):  # still a frozen dataclass
        delegated.means = None  # type: ignore[misc]


def test_trace_attributes_include_means_identifiers_when_set():
    """Acceptance #43: trace attributes include the means identifier(s) when a means is set."""
    ctx = RequestContext(tenant_id=uuid.uuid4(), identity_id=uuid.uuid4())
    autonomous = ctx.acting_through("credential", "cred-123")

    attrs = autonomous.trace_attributes()

    assert attrs["means_kind"] == "credential"
    assert attrs["means_id"] == "cred-123"
    assert attrs["tenant_id"] == str(ctx.tenant_id)
    assert attrs["identity_id"] == str(ctx.identity_id)
    assert attrs["request_id"] == ctx.request_id


def test_existing_actor_only_callers_keep_working_with_and_without_means():
    """Acceptance #43: existing callers that only ever set the actor (no means) keep working
    unchanged -- both the with-means and without-means shapes side by side."""
    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()

    # The shape every existing construction site (app/deps.py, app/mcp/server.py) still uses.
    plain = RequestContext(tenant_id=tenant_id, identity_id=identity_id, roles=frozenset({"admin"}))
    assert plain.means is None
    assert plain.has_role("admin")
    assert set(plain.trace_attributes()) == {"tenant_id", "identity_id", "request_id"}

    with_means = plain.acting_through("agent", "assistant")
    assert with_means.means is not None
    assert with_means.has_role("admin")
    assert set(with_means.trace_attributes()) == {
        "tenant_id",
        "identity_id",
        "request_id",
        "means_kind",
        "means_id",
    }
