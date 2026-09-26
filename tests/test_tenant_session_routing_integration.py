"""Embedded-Postgres integration tests for `tenant_session()` routing through the control plane
and the engine registry (ADR-0002, Spec 10 / #75).

A pooled tenant and a dedicated tenant, both seeded through the shared test support package
(`tests.support.seed_tenant`, issue #96/#97): a dedicated tenant's control-plane bookkeeping row
necessarily lives in the pooled database too (`control.tenants` FK-references `public.tenants`)
-- what these tests prove absent from the pooled database is the tenant's own *data* (a row in
`memberships`), not that bookkeeping stub.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

pgserver = pytest.importorskip("pgserver")

from app.context import RequestContext  # noqa: E402
from app.db.session import TenantSuspendedError, get_engine, tenant_session  # noqa: E402
from app.repositories.memberships import MembershipRepository  # noqa: E402
from tests.support import cluster, environment, seed_membership, seed_tenant  # noqa: E402

# `cluster`/`environment` are imported only so pytest can discover them as fixtures from this
# module's namespace -- referenced only by parameter name in the tests below, never called
# directly.
_ = (cluster, environment)


async def test_pooled_tenant_is_served_from_the_default_instance_and_sees_only_its_own_rows(
    environment,
):
    tenant = await seed_tenant(environment, name="Pooled", roles=["member"])
    other = await seed_tenant(environment, name="Other", roles=["member"])

    async with tenant_session(tenant.ctx("member")) as session:
        assert session.get_bind() is get_engine().sync_engine
        memberships = await MembershipRepository().list_for_tenant(session, tenant.ctx("member"))

    assert {m.identity_id for m in memberships} == set(tenant.identities.values())
    assert not (set(tenant.identities.values()) & set(other.identities.values()))


async def test_dedicated_tenant_is_served_from_its_own_instance_and_sees_only_its_own_rows(
    environment,
):
    tenant = await seed_tenant(
        environment, name="Dedicated", roles=["member"], isolation_tier="dedicated"
    )
    assert tenant.database is not None

    async with tenant_session(tenant.ctx("member")) as session:
        assert session.get_bind() is not get_engine().sync_engine
        memberships = await MembershipRepository().list_for_tenant(session, tenant.ctx("member"))

    assert {m.identity_id for m in memberships} == set(tenant.identities.values())


async def test_dedicated_tenants_data_is_physically_absent_from_the_pooled_database(environment):
    """Not merely policy-hidden: forcing the tenant's own context directly against the pooled
    engine (bypassing routing entirely) still returns zero rows, because the tenant's data was
    never written to the pooled database -- only its control-plane bookkeeping stub was."""
    tenant = await seed_tenant(
        environment, name="Dedicated", roles=["member"], isolation_tier="dedicated"
    )

    async with tenant_session(tenant.ctx("member")):
        pass  # exercises routing once; asserted directly against tenant_session() above

    pooled_engine = get_engine()
    async with pooled_engine.connect() as conn:
        async with conn.begin():
            await conn.execute(
                text("SELECT set_config('app.tenant_id', :tid, true)"),
                {"tid": str(tenant.tenant_id)},
            )
            memberships = (await conn.execute(text("SELECT id FROM memberships"))).scalars().all()
    assert memberships == []


async def test_tenant_session_rejects_a_suspended_tenant_and_unsuspending_restores_it(
    environment,
):
    """Spec 9 / #69, ADR-0010, seam 1: the exact same control-plane read `tenant_session()` makes
    to route a session (this file's other tests) also rejects it, before any session against the
    tenant's data is ever opened, and un-suspending (the operator's `unsuspend` command, stood in
    for here by a direct write) restores it -- with nothing re-provisioned, exactly the same
    routing as before suspension.
    """
    tenant = await seed_tenant(environment, name="Suspended", roles=["member"])

    async with tenant.owner_connection() as conn:
        async with conn.begin():
            await conn.execute(
                text("SELECT set_config('app.tenant_id', :tid, true)"),
                {"tid": str(tenant.tenant_id)},
            )
            await conn.execute(
                text("UPDATE control.tenants SET suspended_at = now() WHERE tenant_id = :tid"),
                {"tid": tenant.tenant_id},
            )

    with pytest.raises(TenantSuspendedError) as exc_info:
        async with tenant_session(tenant.ctx("member")):
            pass
    assert exc_info.value.tenant_id == tenant.tenant_id

    # Un-suspend (direct write, standing in for the operator's `unsuspend` command) and the exact
    # same session-building call now succeeds, routed exactly as it would have been all along.
    async with tenant.owner_connection() as conn:
        async with conn.begin():
            await conn.execute(
                text("SELECT set_config('app.tenant_id', :tid, true)"),
                {"tid": str(tenant.tenant_id)},
            )
            await conn.execute(
                text("UPDATE control.tenants SET suspended_at = NULL WHERE tenant_id = :tid"),
                {"tid": tenant.tenant_id},
            )

    async with tenant_session(tenant.ctx("member")) as session:
        assert session.get_bind() is get_engine().sync_engine
        memberships = await MembershipRepository().list_for_tenant(session, tenant.ctx("member"))
    assert {m.identity_id for m in memberships} == set(tenant.identities.values())


async def test_tenant_with_no_control_plane_row_defaults_to_pooled(environment):
    """Existing callers (test_rls_integration.py, test_db_limits_integration.py) seed tenants
    only in `public.tenants`, never in `control.tenants` -- ADR-0002 defaults every tenant to
    pooled, so this must keep working unmodified. `seed_tenant` itself always writes a
    `control.tenants` row (however minimal), so this one scenario -- no control-plane row at
    all -- is seeded directly, the one case the package's `seed_tenant` does not cover by
    design; `seed_membership` (the package's own seam for "a membership `seed_tenant`'s own
    `roles=` doesn't cover") supplies the identity and membership."""
    tenant_id = uuid.uuid4()
    engine = create_async_engine(environment.superuser_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
                {"id": tenant_id, "name": "NoControlRow"},
            )
    finally:
        await engine.dispose()
    identity_id, _membership_id = await seed_membership(
        environment, tenant_id=tenant_id, role="member"
    )

    ctx = RequestContext(tenant_id=tenant_id, identity_id=identity_id)
    async with tenant_session(ctx) as session:
        assert session.get_bind() is get_engine().sync_engine
        memberships = await MembershipRepository().list_for_tenant(session, ctx)
    assert {m.identity_id for m in memberships} == {identity_id}
