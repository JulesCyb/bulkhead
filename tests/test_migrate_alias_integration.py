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
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.db.engine_registry import POOLED_ALIAS

pgserver = pytest.importorskip("pgserver")

import scripts.migrate as migrate_module  # noqa: E402
from tests.support import cluster, create_database, seed_dedicated_control_row  # noqa: E402

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
    migrate_module.asyncio.run(seed_dedicated_control_row(pooled, alias="tenant-green"))

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
    migrate_module.asyncio.run(seed_dedicated_control_row(pooled, alias="tenant-missing-secret"))

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


def test_an_in_process_migration_run_leaves_existing_loggers_enabled(migrate_env):
    """(#118) `migrations/env.py` configures logging from `alembic.ini` on every run. With the
    stdlib default (`disable_existing_loggers=True`) that silently disabled every logger created
    before the run -- in the operator process, everything logged after
    `ensure_dedicated_database` had provisioned a dedicated tenant's database vanished. A logger
    that exists before an in-process `migrate_alias` call, with its own handler, must still be
    enabled and still deliver records afterwards. (Its own handler, not `caplog`: `fileConfig`
    legitimately replaces the *root* logger's handlers, which is where `caplog` listens.)"""
    import logging

    class _Collect(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.messages: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.messages.append(record.getMessage())

    survivor = logging.getLogger("bulkhead.test.survives_migration")
    survivor.setLevel(logging.INFO)
    collector = _Collect()
    survivor.addHandler(collector)
    try:
        assert not survivor.disabled

        migrate_module.migrate_alias(POOLED_ALIAS)

        assert not survivor.disabled
        survivor.info("still here")
        assert collector.messages == ["still here"]
    finally:
        survivor.removeHandler(collector)


def test_0042_downgrade_then_upgrade_touches_exactly_the_two_delegation_means_columns(
    migrate_env,
):
    """#117 AC: migration 0042 applies cleanly (`migrate_all()` already brings a fresh database
    all the way through it, proving the "fresh database" half); this proves the other half -- a
    real `alembic downgrade -1` from head removes exactly `means_kind`/`means_id` from
    `approval_audit_events` and nothing else, and `upgrade head` restores exactly those two columns
    (an already-at-0041 database upgrading cleanly is the same code path a fresh database's own
    walk through every revision, 0042 included, already exercises)."""
    migrate_module.migrate_all()
    pooled = migrate_env["pooled"]

    config = Config(str(migrate_module._ALEMBIC_INI))
    config.set_main_option("script_location", str(migrate_module._REPO_ROOT / "migrations"))
    config.attributes["migration_database_url"] = pooled.owner_url

    async def _columns() -> set[str]:
        engine = create_async_engine(pooled.owner_url)
        try:
            async with engine.connect() as conn:
                rows = (
                    await conn.execute(
                        text(
                            "SELECT column_name FROM information_schema.columns "
                            "WHERE table_name = 'approval_audit_events'"
                        )
                    )
                ).scalars()
                return set(rows.all())
        finally:
            await engine.dispose()

    before = migrate_module.asyncio.run(_columns())
    assert {"means_kind", "means_id"} <= before

    # An explicit target, not "-1": head moved past 0042 (0043, #82).
    command.downgrade(config, "0041")
    after_downgrade = migrate_module.asyncio.run(_columns())
    assert after_downgrade == before - {"means_kind", "means_id"}

    command.upgrade(config, "head")
    after_upgrade = migrate_module.asyncio.run(_columns())
    assert after_upgrade == before


def test_0043_widens_the_pending_action_status_check_and_downgrade_refuses_new_values(
    cluster, migrate_env
):
    """#82: migration 0043 widens `pending_actions_status_check` to the six lifecycle states; a
    downgrade on an empty table restores exactly the three-value constraint, and `upgrade head`
    widens it again. With a row carrying one of the new values, the downgrade refuses (its
    documented limitation) and leaves the schema at 0043 -- nothing is silently rewritten."""
    migrate_module.migrate_all()
    pooled = migrate_env["pooled"]

    config = Config(str(migrate_module._ALEMBIC_INI))
    config.set_main_option("script_location", str(migrate_module._REPO_ROOT / "migrations"))
    config.attributes["migration_database_url"] = pooled.owner_url

    async def _check_definition() -> str:
        engine = create_async_engine(pooled.superuser_url)
        try:
            async with engine.connect() as conn:
                return (
                    await conn.execute(
                        text(
                            "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                            "WHERE conname = 'pending_actions_status_check'"
                        )
                    )
                ).scalar_one()
        finally:
            await engine.dispose()

    async def _plant_executed_row() -> None:
        # Superuser with FK triggers off: this test is about the CHECK constraint, not the
        # tenant/conversation/membership rows a real pending action hangs off.
        engine = create_async_engine(pooled.superuser_url)
        try:
            async with engine.begin() as conn:
                await conn.execute(text("SET LOCAL session_replication_role = replica"))
                await conn.execute(
                    text(
                        "INSERT INTO pending_actions (tenant_id, conversation_id, tool_name, "
                        "args_hash, tool_call_id, asking_membership_id, status, expires_at) "
                        "VALUES (gen_random_uuid(), 'c', 't', 'h', 'call', gen_random_uuid(), "
                        "'executed', now())"
                    )
                )
        finally:
            await engine.dispose()

    widened = migrate_module.asyncio.run(_check_definition())
    for status in ("pending", "approved", "refused", "expired", "executed", "execution_failed"):
        assert f"'{status}'" in widened

    command.downgrade(config, "0042")
    narrowed = migrate_module.asyncio.run(_check_definition())
    assert "'refused'" in narrowed
    for status in ("expired", "executed", "execution_failed"):
        assert f"'{status}'" not in narrowed

    command.upgrade(config, "head")
    assert migrate_module.asyncio.run(_check_definition()) == widened

    migrate_module.asyncio.run(_plant_executed_row())
    with pytest.raises(Exception, match="cannot downgrade 0043"):
        command.downgrade(config, "0042")
    assert migrate_module.asyncio.run(_alembic_version(pooled.owner_url)) == "0043"
    assert migrate_module.asyncio.run(_check_definition()) == widened
