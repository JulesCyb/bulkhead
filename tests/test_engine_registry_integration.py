"""Embedded-Postgres integration tests for the engine-registry seam (`app/db/engine_registry.py`,
ticket #74). A second database on the same embedded cluster (`tests.support.create_database`)
stands in for "a tenant's dedicated database"; the property under test is routing and caching,
not RLS (RLS is out of scope for this ticket -- see #73/#75).
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import text

from app.db import engine_registry
from app.db import session as db_session
from app.db.engine_registry import _secrets_dir

pgserver = pytest.importorskip("pgserver")

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
