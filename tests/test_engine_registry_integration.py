"""Embedded-Postgres integration tests for the engine-registry seam (`app/db/engine_registry.py`,
ticket #74). Two ephemeral `pgserver` instances stand in for "the pooled database" and "a
tenant's dedicated database"; the property under test is routing and caching, not RLS (RLS is
out of scope for this ticket -- see #73/#75).
"""

from __future__ import annotations

import asyncio
import tempfile
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import text

from app import config
from app.db import engine_registry
from app.db import session as db_session

pgserver = pytest.importorskip("pgserver")


@pytest.fixture(scope="module")
def two_servers():
    """Two independent ephemeral Postgres instances: index 0 stands in for the pooled default,
    index 1 for a tenant's dedicated database."""
    servers = []
    urls = []
    for _ in range(2):
        pgdata = tempfile.mkdtemp(prefix="pgdata-engine-registry-")
        server = pgserver.get_server(pgdata)
        sockdir = parse_qs(urlparse(server.get_uri()).query)["host"][0]
        urls.append(f"postgresql+asyncpg://postgres@/postgres?host={sockdir}")
        servers.append(server)
    yield urls
    for server in servers:
        server.cleanup()


@pytest.fixture
def registry_env(two_servers, tmp_path, monkeypatch):
    pooled_url, dedicated_url = two_servers

    settings = config.Settings(
        database_url=pooled_url,
        environment="test",
        auth_mode="dev-headers",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
    )
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(db_session, "get_settings", lambda: settings)
    monkeypatch.setattr(engine_registry, "get_settings", lambda: settings)
    db_session._engine = None
    db_session._session_factory = None
    engine_registry.reset_registry_for_tests()

    secrets_dir = tmp_path / "tenant-db"
    secrets_dir.mkdir()
    monkeypatch.setenv("TENANT_DB_SECRETS_DIR", str(secrets_dir))

    yield {"pooled_url": pooled_url, "dedicated_url": dedicated_url, "secrets_dir": secrets_dir}

    db_session._engine = None
    db_session._session_factory = None
    engine_registry.reset_registry_for_tests()


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
    secrets_dir = registry_env["secrets_dir"]
    (secrets_dir / "tenant-blue").write_text(registry_env["dedicated_url"])

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
    secrets_dir = registry_env["secrets_dir"]
    (secrets_dir / "tenant-green").write_text(registry_env["dedicated_url"])

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
