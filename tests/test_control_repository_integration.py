"""Embedded-Postgres integration tests for the owner-role side of `ControlRepository` (spec A5 /
#113): `create_tenant_record`, `set_suspended`, `read_gateway_credential_alias`/
`write_gateway_credential_alias`, `enumerate_referenced_aliases`, and `get_record`, plus (code
review 2026-09-26) `get_routing_state`, `enumerate_tenants`, `record_operator_action`, and
`IdentityRepository.upsert` -- one two-tenant test per function, the repository convention
(`tests/test_rls_integration.py`'s control-plane sections,
`tests/test_tenant_table_registry_integration.py`).

Every owner-role function here takes an already-open `AsyncConnection`, exactly as
`app/operator/create.py`/`erase.py`/`suspend.py` open one today -- these tests open that
connection themselves (the owner role, `environment.owner_url`) rather than driving it through
the CLI, since the operator commands are not yet rewired onto this repository (that is #114).
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

pgserver = pytest.importorskip("pgserver")

from app.db.session import control_session, tenant_record_session  # noqa: E402
from app.repositories.control import ControlRepository, IdentityRepository  # noqa: E402
from tests.support import (  # noqa: E402
    cluster,
    environment,
    seed_dedicated_control_row,
    seed_tenant,
)

# `cluster`/`environment` are imported only so pytest can discover them as fixtures from this
# module's namespace -- referenced only by parameter name in the tests below, never called
# directly.
_ = (cluster, environment)


async def _raw_control_row(environment, tenant_id: uuid.UUID) -> dict | None:
    """Reads `tenant_id`'s `control.tenants` row directly, as the cluster's own superuser --
    bypassing RLS entirely, so this never depends on the repository being correct."""
    engine = create_async_engine(environment.superuser_url)
    try:
        async with engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text(
                            "SELECT residency, isolation_tier, database_alias, "
                            "gateway_credential_alias, suspended, suspended_at "
                            "FROM control.tenants WHERE tenant_id = :tid"
                        ),
                        {"tid": tenant_id},
                    )
                )
                .mappings()
                .one_or_none()
            )
    finally:
        await engine.dispose()
    return dict(row) if row is not None else None


async def test_create_tenant_record_writes_both_rows_and_leaves_other_tenants_untouched(
    environment,
):
    other = await seed_tenant(environment, name="Other Co", residency="us", via_operator=False)
    tenant_id = uuid.uuid4()

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            await ControlRepository().create_tenant_record(
                conn,
                tenant_id,
                name="Fresh Co",
                residency="eu",
                isolation_tier="pooled",
                database_alias=None,
            )
    finally:
        await engine.dispose()

    row = await _raw_control_row(environment, tenant_id)
    assert row == {
        "residency": "eu",
        "isolation_tier": "pooled",
        "database_alias": None,
        "gateway_credential_alias": None,
        "suspended": False,
        "suspended_at": None,
    }

    # The public.tenants row was written too, with the given name.
    engine = create_async_engine(environment.superuser_url)
    try:
        async with engine.connect() as conn:
            name = (
                await conn.execute(
                    text("SELECT name FROM tenants WHERE id = :tid"), {"tid": tenant_id}
                )
            ).scalar_one()
    finally:
        await engine.dispose()
    assert name == "Fresh Co"

    # The other, pre-existing tenant's own row is untouched.
    other_row = await _raw_control_row(environment, other.tenant_id)
    assert other_row["residency"] == "us"


async def test_get_record_reads_only_the_requested_tenant_even_on_a_reused_connection(
    environment,
):
    tenant_a = await seed_tenant(environment, name="Record A", residency="eu", via_operator=False)
    tenant_b = await seed_tenant(environment, name="Record B", residency="us", via_operator=False)

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            record_a = await ControlRepository().get_record(conn, tenant_a.tenant_id)
            # Same connection, immediately after reading A: proves the private forced-RLS helper
            # re-binds `app.tenant_id` on every call rather than leaving A's setting in place.
            record_b = await ControlRepository().get_record(conn, tenant_b.tenant_id)
    finally:
        await engine.dispose()

    assert (record_a.tenant_id, record_a.residency) == (tenant_a.tenant_id, "eu")
    assert (record_b.tenant_id, record_b.residency) == (tenant_b.tenant_id, "us")


async def test_get_record_of_an_unknown_tenant_is_the_pooled_default(environment):
    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            record = await ControlRepository().get_record(conn, uuid.uuid4())
    finally:
        await engine.dispose()
    assert record.isolation_tier == "pooled"
    assert record.suspended is False
    assert record.residency is None


async def test_set_suspended_is_idempotent_and_isolated_per_tenant(environment):
    tenant_a = await seed_tenant(environment, name="Suspend A", via_operator=False)
    tenant_b = await seed_tenant(environment, name="Suspend B", via_operator=False)

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            first = await ControlRepository().set_suspended(conn, tenant_a.tenant_id, True)
    finally:
        await engine.dispose()
    assert first.changed is True
    assert first.suspended_at is not None

    # tenant B, never touched, is still active.
    row_b = await _raw_control_row(environment, tenant_b.tenant_id)
    assert row_b["suspended"] is False
    assert row_b["suspended_at"] is None

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            second = await ControlRepository().set_suspended(conn, tenant_a.tenant_id, True)
    finally:
        await engine.dispose()
    assert second.changed is False  # already suspended: a no-op, never an error

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            unsuspended = await ControlRepository().set_suspended(conn, tenant_a.tenant_id, False)
    finally:
        await engine.dispose()
    assert unsuspended.changed is True
    assert unsuspended.suspended_at is None


async def test_gateway_credential_alias_read_and_write_is_isolated_per_tenant(environment):
    tenant_a = await seed_tenant(environment, name="Gateway A", via_operator=False)
    tenant_b = await seed_tenant(environment, name="Gateway B", via_operator=False)

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            before_a = await ControlRepository().read_gateway_credential_alias(
                conn, tenant_a.tenant_id
            )
            before_b = await ControlRepository().read_gateway_credential_alias(
                conn, tenant_b.tenant_id
            )
    finally:
        await engine.dispose()
    assert (before_a, before_b) == (None, None)

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            await ControlRepository().write_gateway_credential_alias(
                conn, tenant_a.tenant_id, "gw-alias-a"
            )
    finally:
        await engine.dispose()

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            alias_a = await ControlRepository().read_gateway_credential_alias(
                conn, tenant_a.tenant_id
            )
            alias_b = await ControlRepository().read_gateway_credential_alias(
                conn, tenant_b.tenant_id
            )
    finally:
        await engine.dispose()
    assert alias_a == "gw-alias-a"
    assert alias_b is None  # tenant B's own row is untouched by tenant A's write

    # Clearing tenant A's alias never affects tenant B's (still unset).
    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            await ControlRepository().write_gateway_credential_alias(conn, tenant_a.tenant_id, None)
            cleared_a = await ControlRepository().read_gateway_credential_alias(
                conn, tenant_a.tenant_id
            )
            still_b = await ControlRepository().read_gateway_credential_alias(
                conn, tenant_b.tenant_id
            )
    finally:
        await engine.dispose()
    assert cleared_a is None
    assert still_b is None


async def test_enumerate_referenced_aliases_reports_pooled_and_every_dedicated_alias(environment):
    await seed_tenant(environment, name="Pooled Only", via_operator=False)
    await seed_dedicated_control_row(environment, alias="tenant-fake-alias")

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.connect() as conn:
            aliases = await ControlRepository().enumerate_referenced_aliases(conn)
    finally:
        await engine.dispose()

    assert set(aliases) >= {"pooled", "tenant-fake-alias"}


async def test_enumerate_referenced_aliases_also_works_from_an_app_role_session(environment):
    """The same method, called from an app-role session with no tenant context (exactly how
    `app/db/guard.py`'s `_referenced_aliases` calls it) rather than an owner-role connection --
    proving the `AsyncConnection | AsyncSession` parameter really does serve both call sites."""
    await seed_dedicated_control_row(environment, alias="tenant-app-role-alias")

    async with control_session() as session:
        aliases = await ControlRepository().enumerate_referenced_aliases(session)

    assert "tenant-app-role-alias" in aliases


# --- Code review 2026-09-26: second-tenant tests for the remaining repository functions ---------


async def _suspend(environment, tenant_id: uuid.UUID) -> None:
    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            await ControlRepository().set_suspended(conn, tenant_id, True)
    finally:
        await engine.dispose()


async def test_get_routing_state_sees_only_the_session_tenants_row(environment):
    tenant_a = await seed_tenant(environment, name="Routing A", via_operator=False)
    tenant_b_id = await seed_dedicated_control_row(environment, alias="tenant-routing-b")
    await _suspend(environment, tenant_b_id)

    async with tenant_record_session(tenant_a.tenant_id) as session:
        state_a = await ControlRepository().get_routing_state(session, tenant_id=tenant_a.tenant_id)
        # Same session, tenant A's context: tenant B's dedicated, suspended row is invisible, so
        # it reads as the pooled default rather than leaking B's alias or suspension.
        b_from_a = await ControlRepository().get_routing_state(session, tenant_id=tenant_b_id)
    async with tenant_record_session(tenant_b_id) as session:
        state_b = await ControlRepository().get_routing_state(session, tenant_id=tenant_b_id)

    assert (state_a.isolation_tier, state_a.suspended_at) == ("pooled", None)
    assert (b_from_a.isolation_tier, b_from_a.database_alias, b_from_a.suspended_at) == (
        None,
        None,
        None,
    )
    assert (state_b.isolation_tier, state_b.database_alias) == ("dedicated", "tenant-routing-b")
    assert state_b.suspended_at is not None


async def test_enumerate_tenants_reports_each_tenants_own_facts_once(environment):
    tenant_a = await seed_tenant(
        environment, name="Enumerate A", residency="eu", via_operator=False
    )
    tenant_b = await seed_tenant(
        environment, name="Enumerate B", residency="us", via_operator=False
    )
    await _suspend(environment, tenant_b.tenant_id)

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.connect() as conn:
            summaries = await ControlRepository().enumerate_tenants(conn)
    finally:
        await engine.dispose()

    ours = [s for s in summaries if s.tenant_id in (tenant_a.tenant_id, tenant_b.tenant_id)]
    assert len(ours) == 2  # each tenant exactly once
    by_id = {s.tenant_id: s for s in ours}
    a, b = by_id[tenant_a.tenant_id], by_id[tenant_b.tenant_id]
    assert (a.name, a.residency, a.suspended, a.suspended_at) == ("Enumerate A", "eu", False, None)
    assert (b.name, b.residency, b.suspended) == ("Enumerate B", "us", True)
    assert b.suspended_at is not None


async def test_record_operator_action_writes_one_row_for_the_named_tenant_only(environment):
    tenant_a = await seed_tenant(environment, name="Audit A", via_operator=False)
    tenant_b = await seed_tenant(environment, name="Audit B", via_operator=False)

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            await ControlRepository().record_operator_action(
                conn,
                tenant_id=tenant_a.tenant_id,
                action="review-test",
                details_json='{"outcome": "ok"}',
            )
    finally:
        await engine.dispose()

    engine = create_async_engine(environment.superuser_url)
    try:
        async with engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT tenant_id, action, details FROM control.operator_actions "
                        "WHERE tenant_id IN (:a, :b)"
                    ),
                    {"a": tenant_a.tenant_id, "b": tenant_b.tenant_id},
                )
            ).all()
    finally:
        await engine.dispose()

    assert [(r.tenant_id, r.action, r.details) for r in rows] == [
        (tenant_a.tenant_id, "review-test", {"outcome": "ok"})
    ]


async def test_identity_upsert_is_idempotent_and_leaves_other_identities_untouched(environment):
    other = await seed_tenant(
        environment, name="Upsert Other", roles=("member",), via_operator=False
    )
    other_identity_id = other.identities["member"]

    async def _identity_rows(where: str, params: dict) -> list[tuple]:
        engine = create_async_engine(environment.superuser_url)
        try:
            async with engine.connect() as conn:
                result = await conn.execute(
                    text(
                        f"SELECT id, issuer, subject, email FROM control.identities WHERE {where}"
                    ),
                    params,
                )
                return [tuple(row) for row in result]
        finally:
            await engine.dispose()

    other_before = await _identity_rows("id = :id", {"id": other_identity_id})
    subject = f"review-{uuid.uuid4()}"

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            first = await IdentityRepository().upsert(
                conn, id=uuid.uuid4(), issuer="review", subject=subject, email="a@example.test"
            )
        async with engine.begin() as conn:
            # A different candidate id for the same (issuer, subject): the existing row wins.
            second = await IdentityRepository().upsert(
                conn, id=uuid.uuid4(), issuer="review", subject=subject, email="a@example.test"
            )
    finally:
        await engine.dispose()

    assert second == first
    assert await _identity_rows(
        "issuer = :issuer AND subject = :subject", {"issuer": "review", "subject": subject}
    ) == [(first, "review", subject, "a@example.test")]
    # The other tenant's member identity is exactly as it was.
    assert await _identity_rows("id = :id", {"id": other_identity_id}) == other_before
