"""Embedded-Postgres integration tests for the owner-role side of the membership repository
(code review 2026-09-26): `app.repositories.memberships.ensure_membership`, the one function the
operator's `create` command (pooled and dedicated alike) writes a first admin membership through
-- and its read-side sibling `get_role_owner` (#83) -- one two-tenant test each, the repository
convention (`tests/test_rls_integration.py`).
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

pgserver = pytest.importorskip("pgserver")

from app.repositories.control import ControlRepository  # noqa: E402
from app.repositories.memberships import ensure_membership, get_role_owner  # noqa: E402
from tests.support import cluster, environment, seed_membership, seed_tenant  # noqa: E402

# Imported only so pytest discovers them as fixtures from this module's namespace.
_ = (cluster, environment)


async def _memberships_of(environment, identity_id: uuid.UUID) -> dict[uuid.UUID, str]:
    """Every (tenant -> role) membership of `identity_id`, read as the cluster's own superuser --
    bypassing RLS, so this never depends on the repository being correct."""
    engine = create_async_engine(environment.superuser_url)
    try:
        async with engine.connect() as conn:
            rows = await conn.execute(
                text("SELECT tenant_id, role FROM memberships WHERE identity_id = :iid"),
                {"iid": identity_id},
            )
            return {row.tenant_id: row.role for row in rows}
    finally:
        await engine.dispose()


async def test_ensure_membership_is_idempotent_and_never_touches_another_tenant(environment):
    tenant_a = await seed_tenant(environment, name="Membership A", via_operator=False)
    tenant_b = await seed_tenant(environment, name="Membership B", via_operator=False)
    # The same identity is already a plain member of tenant B.
    identity_id, _membership_b = await seed_membership(
        environment, tenant_id=tenant_b.tenant_id, role="member"
    )

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            # The pooled path's caller sets the tenant context through the control repository
            # (here `get_record`, in `create` its `get_record`/`create_tenant_record`).
            await ControlRepository().get_record(conn, tenant_a.tenant_id)
            first = await ensure_membership(
                conn, tenant_id=tenant_a.tenant_id, identity_id=identity_id, role="admin"
            )
            second = await ensure_membership(
                conn, tenant_id=tenant_a.tenant_id, identity_id=identity_id, role="admin"
            )
    finally:
        await engine.dispose()

    assert (first, second) == ("created", "already exists")
    # Exactly one new row, in tenant A; tenant B's own membership of the identity is untouched.
    assert await _memberships_of(environment, identity_id) == {
        tenant_a.tenant_id: "admin",
        tenant_b.tenant_id: "member",
    }


async def test_ensure_membership_refuses_a_tenant_other_than_the_connections_context(
    environment,
):
    tenant_a = await seed_tenant(environment, name="Context A", via_operator=False)
    tenant_b = await seed_tenant(environment, name="Context B", via_operator=False)
    identity_id, _membership_b = await seed_membership(
        environment, tenant_id=tenant_b.tenant_id, role="member"
    )

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            await ControlRepository().get_record(conn, tenant_a.tenant_id)
            # Context is tenant A; asking about tenant B's existing membership must fail closed,
            # never report "already exists" or "created" for a tenant it cannot see.
            with pytest.raises(RuntimeError, match="tenant context"):
                await ensure_membership(
                    conn, tenant_id=tenant_b.tenant_id, identity_id=identity_id, role="admin"
                )
    finally:
        await engine.dispose()

    assert await _memberships_of(environment, identity_id) == {tenant_b.tenant_id: "member"}


async def test_get_role_owner_reads_only_the_context_tenants_membership(environment):
    tenant_a = await seed_tenant(environment, name="Role A", via_operator=False)
    tenant_b = await seed_tenant(environment, name="Role B", via_operator=False)
    # The same identity is an admin of tenant A and a plain member of tenant B.
    identity_id, _membership_a = await seed_membership(
        environment, tenant_id=tenant_a.tenant_id, role="admin"
    )
    stranger_id = uuid.uuid4()

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            await ControlRepository().get_record(conn, tenant_b.tenant_id)
            await ensure_membership(
                conn, tenant_id=tenant_b.tenant_id, identity_id=identity_id, role="member"
            )
        async with engine.begin() as conn:
            await ControlRepository().get_record(conn, tenant_a.tenant_id)
            role_in_a = await get_role_owner(
                conn, tenant_id=tenant_a.tenant_id, identity_id=identity_id
            )
            no_membership = await get_role_owner(
                conn, tenant_id=tenant_a.tenant_id, identity_id=stranger_id
            )
            # Context is tenant A: tenant B's membership of the same identity must be unreachable,
            # refused outright rather than answered with tenant B's role.
            with pytest.raises(RuntimeError, match="tenant context"):
                await get_role_owner(conn, tenant_id=tenant_b.tenant_id, identity_id=identity_id)
    finally:
        await engine.dispose()

    assert role_in_a == "admin"
    assert no_membership is None
