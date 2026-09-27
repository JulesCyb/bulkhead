"""Embedded-Postgres integration tests for the engine-registry seam (`app/db/engine_registry.py`,
ticket #74). A second database on the same embedded cluster (`tests.support.create_database`)
stands in for "a tenant's dedicated database"; the property under test is routing and caching,
not RLS (RLS is out of scope for this ticket -- see #73/#75).

Issue #81 extends the property under test: a brand-new dedicated engine is now itself guarded
(`app.db.guard.check_role_and_rls`) before `get_engine_for_alias` ever caches or returns it, so
the tests at the bottom of this file build a real dedicated database with a table missing forced
Row-Level Security -- exactly the shape `tests/test_guard_multi_engine_integration.py` uses for
`run_role_rls_guard` itself -- and drive `get_engine_for_alias` directly, never
`run_role_rls_guard`.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.db import engine_registry
from app.db import guard as guard_module
from app.db import session as db_session
from app.db.engine_registry import _secrets_dir

pgserver = pytest.importorskip("pgserver")

import scripts.migrate as migrate_module  # noqa: E402
from tests.support import cluster, create_database, environment  # noqa: E402

# `cluster`/`environment` are imported only so pytest can discover them as fixtures from this
# module's namespace -- referenced only by parameter name in the tests below, never called
# directly.
_ = (cluster, environment)


@pytest.fixture
async def registry_env(environment):
    """`environment` already points `Settings`/the session layer/the engine registry at the
    pooled cluster and resets every cache before and after -- the cache-clear/engine-reset dance
    this fixture used to repeat by hand. A second database on the same cluster
    (`tests.support.create_database`) stands in for a tenant's dedicated database -- named
    uniquely per test, since `cluster` (and so its underlying server) is session-scoped and
    `CREATE DATABASE` does not tolerate a repeat name."""
    dedicated = await create_database(environment, f"registry-dedicated-{uuid.uuid4().hex[:8]}")
    yield {"pooled_url": environment.app_url, "dedicated_url": dedicated.app_url}


async def _select_1(engine) -> int:
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT 1"))
        return result.scalar_one()


async def test_pooled_alias_is_stable_and_never_reads_a_secret_file(registry_env, monkeypatch):
    def _must_not_be_called(alias):
        raise AssertionError("the pooled alias must never read a tenant-secret file")

    monkeypatch.setattr(engine_registry, "_read_dsn", _must_not_be_called)

    first = await engine_registry.get_engine_for_alias(engine_registry.POOLED_ALIAS)
    second = await engine_registry.get_engine_for_alias(engine_registry.POOLED_ALIAS)

    assert first is second
    assert first is db_session.get_engine()
    assert await _select_1(first) == 1


async def test_dedicated_alias_reads_secret_file_and_serves_a_second_instance(registry_env):
    (_secrets_dir() / "tenant-blue").write_text(registry_env["dedicated_url"])

    dedicated = await engine_registry.get_engine_for_alias("tenant-blue")
    pooled = await engine_registry.get_engine_for_alias(engine_registry.POOLED_ALIAS)

    assert dedicated is not pooled
    assert str(dedicated.url) != str(pooled.url)
    assert await _select_1(dedicated) == 1
    assert await _select_1(pooled) == 1

    # Cached: a second request for the same alias returns the identical engine object.
    again = await engine_registry.get_engine_for_alias("tenant-blue")
    assert again is dedicated


async def test_concurrent_requests_for_a_new_alias_build_exactly_one_engine(
    registry_env, monkeypatch
):
    (_secrets_dir() / "tenant-green").write_text(registry_env["dedicated_url"])

    build_calls: list[str] = []
    original_build = engine_registry._build_engine

    def counting_build(dsn: str):
        build_calls.append(dsn)
        return original_build(dsn)

    monkeypatch.setattr(engine_registry, "_build_engine", counting_build)

    results = await asyncio.gather(
        *(engine_registry.get_engine_for_alias("tenant-green") for _ in range(25))
    )

    assert len(build_calls) == 1
    assert len({id(engine) for engine in results}) == 1
    assert await _select_1(results[0]) == 1


async def test_unknown_alias_raises_instead_of_falling_back_to_pooled(registry_env):
    with pytest.raises(engine_registry.UnknownDatabaseAliasError):
        await engine_registry.get_engine_for_alias("no-such-alias")

    # The pooled alias still resolves normally -- the failed lookup left no bad cache entry
    # and did not disturb the pooled path.
    pooled = await engine_registry.get_engine_for_alias(engine_registry.POOLED_ALIAS)
    assert await _select_1(pooled) == 1
    assert "no-such-alias" not in engine_registry._engines


# --- issue #81: a brand-new dedicated engine is guarded before it is cached or returned ---


async def test_dedicated_alias_with_a_table_missing_forced_rls_is_refused_and_not_cached(
    environment,
):
    """Acceptance (#81): a dedicated alias whose database has a table without forced Row-Level
    Security is refused on the very first `get_engine_for_alias` call -- not only on the next
    `/ready` probe or `run_role_rls_guard()` sweep -- and the failing engine is never cached. A
    second call re-runs the check from scratch rather than serving anything cached; once the
    table is fixed, the alias becomes healthy, cached, and stable."""
    from app.db.models import TENANT_ISOLATION_EXCEPTIONS

    assert "scratch_unforced" not in TENANT_ISOLATION_EXCEPTIONS

    alias = f"registry-unforced-{uuid.uuid4().hex[:8]}"
    dedicated = await create_database(environment, alias)

    (migrate_module._migrations_secrets_dir() / alias).write_text(dedicated.owner_url)
    await asyncio.to_thread(migrate_module.migrate_alias, alias)
    (_secrets_dir() / alias).write_text(dedicated.app_url)

    admin_engine = create_async_engine(dedicated.owner_url)
    try:
        async with admin_engine.begin() as conn:
            await conn.execute(text("CREATE TABLE scratch_unforced (id int)"))

        with pytest.raises(guard_module.PrivilegedRoleOrMissingRLSError):
            await engine_registry.get_engine_for_alias(alias)
        assert alias not in engine_registry._engines

        # A later call retries from scratch: still non-compliant, still refused, still uncached
        # -- never a cached bad engine served on a second try.
        with pytest.raises(guard_module.PrivilegedRoleOrMissingRLSError):
            await engine_registry.get_engine_for_alias(alias)
        assert alias not in engine_registry._engines

        async with admin_engine.begin() as conn:
            await conn.execute(text("DROP TABLE scratch_unforced"))
    finally:
        await admin_engine.dispose()

    # Now compliant: the alias builds cleanly, is cached, and a second call returns the exact
    # same engine instance -- a healthy alias is cached and returned, unlike a failing one.
    healthy = await engine_registry.get_engine_for_alias(alias)
    again = await engine_registry.get_engine_for_alias(alias)
    assert healthy is again
    assert await _select_1(healthy) == 1


async def test_dedicated_alias_connected_as_a_bypassrls_role_is_refused_and_not_cached(
    environment,
):
    """Acceptance (#81), the other half of `check_role_and_rls`: a dedicated alias whose secret
    file names a privileged (BYPASSRLS) role is refused on first use and never cached, exactly
    like the missing-forced-RLS case above."""
    alias = f"registry-bypassrls-{uuid.uuid4().hex[:8]}"
    dedicated = await create_database(environment, alias)

    (migrate_module._migrations_secrets_dir() / alias).write_text(dedicated.owner_url)
    await asyncio.to_thread(migrate_module.migrate_alias, alias)
    # The dedicated database's own bootstrap superuser stands in for a misconfigured deployment
    # (the same substitution `test_guard_multi_engine_integration.py` uses for
    # `run_role_rls_guard`).
    (_secrets_dir() / alias).write_text(dedicated.superuser_url)

    with pytest.raises(guard_module.PrivilegedRoleOrMissingRLSError):
        await engine_registry.get_engine_for_alias(alias)
    assert alias not in engine_registry._engines
