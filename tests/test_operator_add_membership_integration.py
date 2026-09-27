"""Embedded-Postgres integration tests for the `add-membership` command (issue #83, ADR-0003,
ADR-0010, ADR-0012): attaching an additional membership to an already-provisioned tenant.

Pattern: `tests/test_operator_tool_integration.py` (audited dispatch through
`app.operator.cli.run_operator`, never its private dispatch internals). Seeding comes from
`tests.support` (issue #96 / spec #90, "A6"); a pooled tenant is seeded through the operator's own
`create` command (`seed_tenant`'s default, `via_operator=True`) so `add-membership` is exercised
against exactly the rows `create` itself would have produced.
"""

from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

pgserver = pytest.importorskip("pgserver")

from tests.support import cluster, environment, seed_tenant  # noqa: E402

# `cluster`/`environment` are imported only so pytest can discover them as fixtures from this
# module's namespace (the same pattern `tests/test_operator_tool_integration.py` uses) --
# referenced only by parameter name in the tests below, never called directly.
_ = (cluster, environment)


async def _operator_actions(superuser_url: str, action: str) -> list:
    """Same helper as `tests/test_operator_tool_integration.py`'s own private copy -- only the
    superuser can read `control.operator_actions` (`app_owner` itself holds INSERT-only on it by
    grant, migration 0004)."""
    engine = create_async_engine(superuser_url)
    try:
        async with engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT tenant_id, action, details, performed_at "
                        "FROM control.operator_actions WHERE action = :action "
                        "ORDER BY performed_at"
                    ),
                    {"action": action},
                )
            ).all()
    finally:
        await engine.dispose()
    return [
        (
            tenant_id,
            action,
            json.loads(details) if isinstance(details, str) else details,
            performed_at,
        )
        for tenant_id, action, details, performed_at in rows
    ]


async def _identity_count(superuser_url: str, *, issuer: str, subject: str) -> int:
    engine = create_async_engine(superuser_url)
    try:
        async with engine.connect() as conn:
            return (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM control.identities "
                        "WHERE issuer = :issuer AND subject = :subject"
                    ),
                    {"issuer": issuer, "subject": subject},
                )
            ).scalar_one()
    finally:
        await engine.dispose()


async def _membership_role(superuser_url: str, *, tenant_id: uuid.UUID, identity_id: uuid.UUID):
    engine = create_async_engine(superuser_url)
    try:
        async with engine.connect() as conn:
            return (
                await conn.execute(
                    text(
                        "SELECT role FROM memberships WHERE tenant_id = :tid AND identity_id = :iid"
                    ),
                    {"tid": tenant_id, "iid": identity_id},
                )
            ).scalar_one_or_none()
    finally:
        await engine.dispose()


async def _membership_count(superuser_url: str, *, tenant_id: uuid.UUID, identity_id: uuid.UUID):
    engine = create_async_engine(superuser_url)
    try:
        async with engine.connect() as conn:
            return (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM memberships "
                        "WHERE tenant_id = :tid AND identity_id = :iid"
                    ),
                    {"tid": tenant_id, "iid": identity_id},
                )
            ).scalar_one()
    finally:
        await engine.dispose()


async def test_add_membership_creates_new_identity_and_membership(environment, capsys):
    """Seam 1: a fresh identity is created, its membership is created with the requested role,
    and the invocation is audited with the real tenant id."""
    from app.operator.cli import run_operator

    tenant = await seed_tenant(environment, name="Add Membership Co", residency="eu")

    engine = create_async_engine(environment.owner_url)
    exit_code = await run_operator(
        ["add-membership", str(tenant.tenant_id), "member", "new-member@example.test"],
        engine=engine,
    )
    await engine.dispose()
    assert exit_code == 0

    assert (
        await _identity_count(
            environment.superuser_url, issuer="dev-seed", subject="new-member@example.test"
        )
        == 1
    )

    engine = create_async_engine(environment.superuser_url)
    try:
        async with engine.connect() as conn:
            identity_id = (
                await conn.execute(
                    text(
                        "SELECT id FROM control.identities "
                        "WHERE issuer = 'dev-seed' AND subject = 'new-member@example.test'"
                    )
                )
            ).scalar_one()
    finally:
        await engine.dispose()

    role = await _membership_role(
        environment.superuser_url, tenant_id=tenant.tenant_id, identity_id=identity_id
    )
    assert role == "member"

    rows = await _operator_actions(environment.superuser_url, "add-membership")
    assert len(rows) >= 1
    tenant_id_logged, action, details, performed_at = rows[-1]
    assert str(tenant_id_logged) == str(tenant.tenant_id)
    assert action == "add-membership"
    assert details["outcome"].startswith("ok: ")
    assert performed_at is not None

    assert f"{identity_id}" in capsys.readouterr().out


async def test_add_membership_reuses_existing_identity_no_duplicate_row(environment):
    """Seam 2: an identity already known by (issuer, subject) -- here, one seeded as another
    tenant's own admin -- is reused rather than duplicated, and the new tenant gets a fresh
    membership for it."""
    from app.operator.cli import run_operator

    home_tenant = await seed_tenant(
        environment, name="Identity Home Co", admin_email="shared-person@example.test"
    )
    other_tenant = await seed_tenant(environment, name="Identity Second Co")

    assert (
        await _identity_count(
            environment.superuser_url, issuer="dev-seed", subject="shared-person@example.test"
        )
        == 1
    )

    engine = create_async_engine(environment.owner_url)
    exit_code = await run_operator(
        ["add-membership", str(other_tenant.tenant_id), "member", "shared-person@example.test"],
        engine=engine,
    )
    await engine.dispose()
    assert exit_code == 0

    # Still exactly one identity row for this (issuer, subject) -- no duplicate.
    assert (
        await _identity_count(
            environment.superuser_url, issuer="dev-seed", subject="shared-person@example.test"
        )
        == 1
    )

    shared_identity_id = home_tenant.identities["admin"]
    role_in_other_tenant = await _membership_role(
        environment.superuser_url, tenant_id=other_tenant.tenant_id, identity_id=shared_identity_id
    )
    assert role_in_other_tenant == "member"

    # The identity's original membership (home_tenant, admin) is untouched.
    role_in_home_tenant = await _membership_role(
        environment.superuser_url, tenant_id=home_tenant.tenant_id, identity_id=shared_identity_id
    )
    assert role_in_home_tenant == "admin"


async def test_add_membership_rerun_is_a_no_op(environment, capsys):
    """Seam 3: re-running the exact same invocation reports "already exists" and leaves exactly
    one membership row -- but is still audited a second time."""
    from app.operator.cli import run_operator

    tenant = await seed_tenant(environment, name="Rerun Co")
    argv = ["add-membership", str(tenant.tenant_id), "support", "rerun@example.test"]

    first_engine = create_async_engine(environment.owner_url)
    assert await run_operator(argv, engine=first_engine) == 0
    await first_engine.dispose()
    first_out = capsys.readouterr().out
    assert "created" in first_out

    second_engine = create_async_engine(environment.owner_url)
    assert await run_operator(argv, engine=second_engine) == 0
    await second_engine.dispose()
    second_out = capsys.readouterr().out
    assert "already exists" in second_out

    engine = create_async_engine(environment.superuser_url)
    try:
        async with engine.connect() as conn:
            identity_id = (
                await conn.execute(
                    text(
                        "SELECT id FROM control.identities "
                        "WHERE issuer = 'dev-seed' AND subject = 'rerun@example.test'"
                    )
                )
            ).scalar_one()
    finally:
        await engine.dispose()

    count = await _membership_count(
        environment.superuser_url, tenant_id=tenant.tenant_id, identity_id=identity_id
    )
    assert count == 1

    rows = await _operator_actions(environment.superuser_url, "add-membership")
    matching = [r for r in rows if str(r[0]) == str(tenant.tenant_id)]
    assert len(matching) >= 2
    assert matching[-2][2]["outcome"].startswith("ok: ")
    assert matching[-1][2]["outcome"].startswith("ok: ")


async def test_add_membership_rejects_invalid_role_before_any_write(environment):
    """Seam 4a: an unrecognized role is rejected before any write -- no identity row, no
    membership row -- called directly (bypassing argparse's own `choices=` gate, which would
    otherwise reject it first)."""
    from app.operator.add_membership import UnrecognizedRoleError, add_membership

    tenant = await seed_tenant(environment, name="Invalid Role Co")

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            with pytest.raises(UnrecognizedRoleError):
                await add_membership(
                    conn,
                    str(tenant.tenant_id),
                    role="superadmin",
                    email="nobody@example.test",
                )
    finally:
        await engine.dispose()

    assert (
        await _identity_count(
            environment.superuser_url, issuer="dev-seed", subject="nobody@example.test"
        )
        == 0
    )


async def test_add_membership_unknown_tenant_not_found(environment):
    """Seam 4b: an unresolvable tenant fails exactly like `suspend`'s own lookup failure --
    `TenantNotFoundError`, exit code 1, the failure audited under the sentinel tenant id."""
    from app.operator.cli import run_operator

    unknown_id = str(uuid.uuid4())
    engine = create_async_engine(environment.owner_url)
    exit_code = await run_operator(
        ["add-membership", unknown_id, "member", "nobody@example.test"], engine=engine
    )
    await engine.dispose()
    assert exit_code == 1

    rows = await _operator_actions(environment.superuser_url, "add-membership")
    _, _, details, _ = rows[-1]
    assert details["outcome"] == "error"
    assert "TenantNotFoundError" in details["error"]

    assert (
        await _identity_count(
            environment.superuser_url, issuer="dev-seed", subject="nobody@example.test"
        )
        == 0
    )


async def test_add_membership_refuses_a_role_change_on_an_existing_membership(environment):
    """Seam 5: an existing membership with a role different from the one requested is refused,
    unchanged -- never silently updated."""
    from app.operator.cli import run_operator

    tenant = await seed_tenant(environment, name="Role Conflict Co")

    first_engine = create_async_engine(environment.owner_url)
    assert (
        await run_operator(
            ["add-membership", str(tenant.tenant_id), "member", "conflict@example.test"],
            engine=first_engine,
        )
        == 0
    )
    await first_engine.dispose()

    second_engine = create_async_engine(environment.owner_url)
    exit_code = await run_operator(
        ["add-membership", str(tenant.tenant_id), "support", "conflict@example.test"],
        engine=second_engine,
    )
    await second_engine.dispose()
    assert exit_code == 1

    rows = await _operator_actions(environment.superuser_url, "add-membership")
    matching = [r for r in rows if str(r[0]) == str(tenant.tenant_id)]
    assert matching[-1][2]["outcome"] == "error"
    assert "MembershipRoleConflictError" in matching[-1][2]["error"]

    engine = create_async_engine(environment.superuser_url)
    try:
        async with engine.connect() as conn:
            identity_id = (
                await conn.execute(
                    text(
                        "SELECT id FROM control.identities "
                        "WHERE issuer = 'dev-seed' AND subject = 'conflict@example.test'"
                    )
                )
            ).scalar_one()
    finally:
        await engine.dispose()

    role = await _membership_role(
        environment.superuser_url, tenant_id=tenant.tenant_id, identity_id=identity_id
    )
    assert role == "member"
    count = await _membership_count(
        environment.superuser_url, tenant_id=tenant.tenant_id, identity_id=identity_id
    )
    assert count == 1


async def test_add_membership_refuses_a_suspended_tenant(environment):
    """Seam 6 (ADR-0010, CONTEXT.md's "Suspension" glossary entry): nothing is provisioned for a
    suspended tenant -- `add-membership` refuses before touching the identity or the
    membership."""
    from app.operator.cli import run_operator

    tenant = await seed_tenant(environment, name="Suspended Add Co")
    await tenant.suspend()

    engine = create_async_engine(environment.owner_url)
    exit_code = await run_operator(
        ["add-membership", str(tenant.tenant_id), "member", "blocked@example.test"],
        engine=engine,
    )
    await engine.dispose()
    assert exit_code == 1

    rows = await _operator_actions(environment.superuser_url, "add-membership")
    matching = [r for r in rows if str(r[0]) == str(tenant.tenant_id)]
    assert matching[-1][2]["outcome"] == "error"
    assert "TenantSuspendedError" in matching[-1][2]["error"]

    assert (
        await _identity_count(
            environment.superuser_url, issuer="dev-seed", subject="blocked@example.test"
        )
        == 0
    )


async def test_add_membership_on_a_dedicated_tenant_lands_in_its_own_database(environment):
    """Seam 7 (ADR-0002): a dedicated tenant's added membership is written into its own physical
    database, never the pooled one -- reusing the same dedicated-tier fixture
    `tests/test_operator_create_dedicated_integration.py`'s tests share
    (`seed_tenant(isolation_tier="dedicated")`, which provisions the database through the same
    `create_tenant` code path `add-membership` itself reuses)."""
    from app.operator.cli import run_operator

    tenant = await seed_tenant(
        environment, name="Dedicated Add Co", residency="us", isolation_tier="dedicated"
    )
    assert tenant.database is not None

    engine = create_async_engine(environment.owner_url)
    exit_code = await run_operator(
        ["add-membership", str(tenant.tenant_id), "member", "dedicated-member@example.test"],
        engine=engine,
    )
    await engine.dispose()
    assert exit_code == 0

    # The pooled database never sees this membership -- only the control-plane bookkeeping for
    # this tenant lives there.
    pooled_engine = create_async_engine(environment.superuser_url)
    try:
        async with pooled_engine.connect() as conn:
            pooled_count = (
                await conn.execute(
                    text("SELECT count(*) FROM memberships WHERE tenant_id = :tid"),
                    {"tid": tenant.tenant_id},
                )
            ).scalar_one()
    finally:
        await pooled_engine.dispose()
    assert pooled_count == 0  # a dedicated tenant's memberships never live in the pooled database

    dedicated_engine = create_async_engine(tenant.database.superuser_url)
    try:
        async with dedicated_engine.connect() as conn:
            row = (
                await conn.execute(
                    text(
                        "SELECT m.role FROM memberships m "
                        "JOIN control.identities i ON i.id = m.identity_id "
                        "WHERE i.issuer = 'dev-seed' AND "
                        "i.subject = 'dedicated-member@example.test'"
                    )
                )
            ).scalar_one_or_none()
    finally:
        await dedicated_engine.dispose()
    assert row == "member"
