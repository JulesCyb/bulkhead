"""Embedded-Postgres integration tests for `scripts/migrate.py` (Spec 10 / #76): running the
current schema against one named database alias, or against every alias the control plane
currently enumerates. Two fresh, unmigrated databases on the shared embedded cluster
(`tests.support.create_database`) stand in for "the pooled database" and a tenant's dedicated
database -- the property under test is which instance(s) a migration run actually touches, not
RLS itself (that is #73/#75's territory), and deliberately *not* the package's own `cluster`
fixture's default database, which is already migrated to head by the time any test runs.
"""

from __future__ import annotations

import uuid

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.db.engine_registry import POOLED_ALIAS

pgserver = pytest.importorskip("pgserver")

import scripts.migrate as migrate_module  # noqa: E402
from tests.support import cluster, create_database  # noqa: E402

# `cluster` is imported only so pytest can discover it as a fixture from this module's
# namespace -- referenced only by parameter name below, never called directly.
_ = cluster


@pytest.fixture
async def pooled(cluster):
    """A fresh, unmigrated database on the shared cluster, standing in for the pooled alias."""
    return await create_database(cluster, f"migrate-pooled-{uuid.uuid4().hex[:8]}")


@pytest.fixture
async def dedicated(cluster):
    """A fresh, unmigrated database on the shared cluster, standing in for a dedicated alias."""
    return await create_database(cluster, f"migrate-dedicated-{uuid.uuid4().hex[:8]}")


class _FakeSecret:
    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value


class _FakeMigrationSettings:
    def __init__(self, url: str) -> None:
        self.database_url_migrations = _FakeSecret(url)


@pytest.fixture
def migrate_env(pooled, monkeypatch, tmp_path):
    """Points scripts.migrate at `pooled` for the pooled alias, and at an empty (until a test
    populates it) directory for dedicated-alias owner-role secrets."""
    monkeypatch.setattr(
        migrate_module,
        "get_migration_settings",
        lambda: _FakeMigrationSettings(pooled.owner_url),
    )
    secrets_dir = tmp_path / "tenant-db-migrations"
    secrets_dir.mkdir()
    monkeypatch.setenv("TENANT_DB_MIGRATIONS_SECRETS_DIR", str(secrets_dir))
    return {"pooled": pooled, "secrets_dir": secrets_dir}


def _head_revision() -> str:
    config = Config(str(migrate_module._ALEMBIC_INI))
    config.set_main_option("script_location", str(migrate_module._REPO_ROOT / "migrations"))
    return ScriptDirectory.from_config(config).get_current_head()


async def _alembic_version(url: str) -> str | None:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            has_table = (
                await conn.execute(text("SELECT to_regclass('public.alembic_version') IS NOT NULL"))
            ).scalar_one()
            if not has_table:
                return None
            return (
                await conn.execute(text("SELECT version_num FROM alembic_version"))
            ).scalar_one()
    finally:
        await engine.dispose()


async def _insert_dedicated_tenant(superuser_url: str, alias: str) -> None:
    """Seeds one control-plane tenant row on an *already-migrated* pooled database, marking it
    dedicated with the given alias -- exactly what `control.database_aliases` enumerates."""
    tenant_id = uuid.uuid4()
    engine = create_async_engine(superuser_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
                {"id": tenant_id, "name": f"dedicated-{alias}"},
            )
            await conn.execute(
                text(
                    "INSERT INTO control.tenants (tenant_id, isolation_tier, database_alias) "
                    "VALUES (:tid, 'dedicated', :alias)"
                ),
                {"tid": tenant_id, "alias": alias},
            )
    finally:
        await engine.dispose()


def test_migrate_all_brings_pooled_alias_to_head_with_no_dedicated_tenant(migrate_env):
    """(#76) With no dedicated tenant recorded, `migrate_all()` (no argument at the CLI) brings
    exactly the pooled default to head."""
    migrate_module.migrate_all()

    version = migrate_module.asyncio.run(_alembic_version(migrate_env["pooled"].owner_url))
    assert version == _head_revision()


def test_explicit_alias_migrates_only_that_instance(migrate_env, dedicated):
    """(#76) Naming a second alias explicitly migrates only that instance, leaving the pooled
    instance's migration version unchanged."""
    pooled = migrate_env["pooled"]
    migrate_module.migrate_all()  # bring the pooled database to head first, as a real deploy would
    pooled_version_before = migrate_module.asyncio.run(_alembic_version(pooled.owner_url))

    secret_path = migrate_env["secrets_dir"] / "tenant-blue"
    secret_path.write_text(dedicated.owner_url)

    migrate_module.migrate_alias("tenant-blue")

    dedicated_version = migrate_module.asyncio.run(_alembic_version(dedicated.owner_url))
    pooled_version_after = migrate_module.asyncio.run(_alembic_version(pooled.owner_url))

    assert dedicated_version == _head_revision()
    assert pooled_version_after == pooled_version_before == _head_revision()


def test_migrate_all_discovers_and_migrates_a_dedicated_alias_from_the_control_plane(
    migrate_env, dedicated
):
    """(#76) After a dedicated alias is seeded in the control plane and its secret file exists,
    invoking the runner with no argument migrates it too -- not only the pooled default."""
    pooled = migrate_env["pooled"]
    migrate_module.migrate_all()
    migrate_module.asyncio.run(_insert_dedicated_tenant(pooled.superuser_url, "tenant-green"))

    secret_path = migrate_env["secrets_dir"] / "tenant-green"
    secret_path.write_text(dedicated.owner_url)

    migrate_module.migrate_all()

    dedicated_version = migrate_module.asyncio.run(_alembic_version(dedicated.owner_url))
    assert dedicated_version == _head_revision()


def test_alias_in_control_plane_without_secret_file_fails_loudly(migrate_env):
    """(#76) An alias the control plane names but with no matching migration-secret file raises
    a clear error instead of being silently skipped -- whether asked for directly or discovered
    through `migrate_all()`."""
    pooled = migrate_env["pooled"]
    migrate_module.migrate_all()
    migrate_module.asyncio.run(
        _insert_dedicated_tenant(pooled.superuser_url, "tenant-missing-secret")
    )

    with pytest.raises(migrate_module.MissingMigrationSecretError) as excinfo:
        migrate_module.migrate_alias("tenant-missing-secret")
    assert "tenant-missing-secret" in str(excinfo.value)

    with pytest.raises(migrate_module.MissingMigrationSecretError):
        migrate_module.migrate_all()


def test_pooled_alias_never_reads_the_migration_secrets_directory(migrate_env):
    """The pooled alias always resolves via DATABASE_URL_MIGRATIONS -- never a file under
    TENANT_DB_MIGRATIONS_SECRETS_DIR, even if one happened to exist named `pooled`."""
    (migrate_env["secrets_dir"] / POOLED_ALIAS).write_text("postgresql+asyncpg://bogus/bogus")

    migrate_module.migrate_all()

    version = migrate_module.asyncio.run(_alembic_version(migrate_env["pooled"].owner_url))
    assert version == _head_revision()
