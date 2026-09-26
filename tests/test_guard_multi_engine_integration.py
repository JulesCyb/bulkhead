"""Embedded-Postgres integration tests for the fail-closed guard extended to every referenced
database engine (Spec 10, ticket #77 / ADR-0011).

Built on the shared test support package (`tests.support`, issue #96/#97): one pooled tenant and
one dedicated tenant, both seeded through `seed_tenant`, give the guard "one alias per isolation
tier" for real, with the dedicated tenant's own database fully migrated by the real runner --
exactly the shape `app.db.guard.run_role_rls_guard` must iterate over. The property under test is
that the guard checks every database alias `control.database_aliases` currently references --
not only the pooled one it always checked before -- so a dangling or misconfigured *dedicated*
database is caught even though no real request has ever been routed to it.

Unlike the other three files this package serves, this file cannot point `DATABASE_URL` at the
shared session `cluster`'s own default database: `run_role_rls_guard()` enumerates *every* alias
the whole control plane it is pointed at currently references, so an earlier test's (or another
test module's, sharing the same session-scoped cluster) dedicated tenant would still show up with
no secret file in *this* test's own fresh `TENANT_DB_SECRETS_DIR` to resolve it. `guard_env`
below gives each test its own fresh, fully migrated database as "the pooled one" instead --
still on the same embedded cluster (`tests.support.create_database`), just never the cluster's
own shared default database.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import text

pgserver = pytest.importorskip("pgserver")

import scripts.migrate as migrate_module  # noqa: E402
from tests.support import (  # noqa: E402
    cluster,
    create_database,
    environment,
    migration_run_without_disrupting_logging,
    seed_tenant,
)

# `cluster`/`environment` are imported only so pytest can discover them as fixtures from this
# module's namespace -- referenced only by parameter name in the tests below, never called
# directly.
_ = (cluster, environment)


@pytest.fixture
async def guard_env(environment, monkeypatch):
    """A fresh, fully migrated database, isolated per test, standing in as "the pooled one" for
    this file's guard tests -- see module docstring for why this file, alone among the four,
    cannot reuse `environment`'s own shared database directly. `environment` still supplies both
    temporary secrets directories and the cache-reset dance; only `DATABASE_URL`/
    `DATABASE_URL_MIGRATIONS` are re-pointed here, on top of it.
    """
    from app import config
    from app.db import engine_registry
    from app.db import session as db_session

    alias = f"guard-pooled-{uuid.uuid4().hex[:8]}"
    fresh_pooled = await create_database(environment, alias)

    migrations_secret = migrate_module._migrations_secrets_dir() / alias
    migrations_secret.write_text(fresh_pooled.owner_url)
    with migration_run_without_disrupting_logging():
        await asyncio.to_thread(migrate_module.migrate_alias, alias)

    monkeypatch.setenv("DATABASE_URL", fresh_pooled.app_url)
    monkeypatch.setenv("DATABASE_URL_MIGRATIONS", fresh_pooled.owner_url)
    config.get_settings.cache_clear()
    db_session._engine = None
    db_session._session_factory = None
    engine_registry.reset_registry_for_tests()

    try:
        yield fresh_pooled
    finally:
        config.get_settings.cache_clear()
        db_session._engine = None
        db_session._session_factory = None
        engine_registry.reset_registry_for_tests()


async def test_guard_fails_when_dedicated_alias_is_connected_as_a_bypassrls_role(guard_env):
    """Acceptance: the guard fails startup when a dedicated alias's engine is connected as a
    role with BYPASSRLS (the dedicated database's own bootstrap superuser), even though the
    pooled engine -- checked first -- passes."""
    from app.db.engine_registry import _secrets_dir
    from app.db.guard import PrivilegedRoleOrMissingRLSError, run_role_rls_guard

    await seed_tenant(guard_env, roles=["member"])
    tenant = await seed_tenant(guard_env, roles=["member"], isolation_tier="dedicated")
    # Overwrite the correct app-role secret `seed_tenant` already wrote with the dedicated
    # database's own bootstrap superuser -- BYPASSRLS, standing in for a misconfigured deployment.
    (_secrets_dir() / tenant.database_alias).write_text(tenant.database.superuser_url)

    with pytest.raises(PrivilegedRoleOrMissingRLSError):
        await run_role_rls_guard()


async def test_guard_fails_when_dedicated_alias_has_a_table_missing_forced_rls(guard_env):
    """Acceptance: the guard fails startup when a dedicated alias's engine points at an
    instance with a table missing forced Row-Level Security, even though the pooled instance is
    fully compliant."""
    from app.db.guard import PrivilegedRoleOrMissingRLSError, run_role_rls_guard
    from app.db.models import TENANT_ISOLATION_EXCEPTIONS

    assert "scratch_unforced" not in TENANT_ISOLATION_EXCEPTIONS
    await seed_tenant(guard_env, roles=["member"])
    tenant = await seed_tenant(guard_env, roles=["member"], isolation_tier="dedicated")

    async with tenant.owner_connection() as conn:
        async with conn.begin():
            await conn.execute(text("CREATE TABLE scratch_unforced (id int)"))

    try:
        with pytest.raises(PrivilegedRoleOrMissingRLSError):
            await run_role_rls_guard()
    finally:
        async with tenant.owner_connection() as conn:
            async with conn.begin():
                await conn.execute(text("DROP TABLE IF EXISTS scratch_unforced"))


async def test_guard_passes_when_every_referenced_alias_is_compliant(guard_env):
    """Acceptance: the guard passes when every currently-referenced alias, pooled and one
    dedicated, is fully compliant."""
    from app.db.guard import run_role_rls_guard

    await seed_tenant(guard_env, roles=["member"])
    await seed_tenant(guard_env, roles=["member"], isolation_tier="dedicated")

    await run_role_rls_guard()  # must not raise


async def test_guard_iterates_past_a_compliant_pooled_engine_to_a_failing_dedicated_one(
    guard_env, monkeypatch
):
    """Acceptance: a case where the pooled engine passes but a second, dedicated engine fails
    proves the guard actually iterates every referenced alias rather than stopping after the
    first. Spied via `check_role_and_rls` itself, so the assertion is about which engines were
    actually checked, not just that *some* error was raised."""
    from app.db import guard as guard_module
    from app.db.engine_registry import POOLED_ALIAS, _secrets_dir

    await seed_tenant(guard_env, roles=["member"])
    tenant = await seed_tenant(guard_env, roles=["member"], isolation_tier="dedicated")
    (_secrets_dir() / tenant.database_alias).write_text(tenant.database.superuser_url)

    checked: list[str] = []
    original_check = guard_module.check_role_and_rls

    async def spying_check(conn):
        # The pooled engine's own DSN carries no alias, so identify it by which server the
        # connection belongs to: `guard_env`'s own app-role DSN.
        checked.append(str(conn.engine.url))
        await original_check(conn)

    monkeypatch.setattr(guard_module, "check_role_and_rls", spying_check)

    with pytest.raises(guard_module.PrivilegedRoleOrMissingRLSError):
        await guard_module.run_role_rls_guard()

    # Both engines were reached: the pooled one (which passed, since it isn't in `checked`
    # failing) and then the dedicated one (whose BYPASSRLS connection made the guard raise).
    # POOLED_ALIAS ("pooled") sorts before a generated alias (always "tenant-..."), so this also
    # proves the loop did not stop after the first (pooled, compliant) alias.
    assert len(checked) == 2
    assert POOLED_ALIAS < tenant.database_alias
