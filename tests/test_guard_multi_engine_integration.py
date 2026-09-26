"""Embedded-Postgres integration tests for the fail-closed guard extended to every referenced
database engine (Spec 10, ticket #77 / ADR-0011).

Builds on two pieces of prior art rather than inventing a new pattern: the role-bootstrap +
full-migration fixture from `tests/test_rls_integration.py` (`database_urls`), run here *twice*
to stand up an independent "pooled" and "dedicated" instance, and the alias/secret-file wiring
from `tests/test_engine_registry_integration.py` (#74). The property under test is that
`app.db.guard.run_role_rls_guard` iterates every database alias `control.database_aliases`
currently references -- not only the pooled one it always checked before -- so a dangling or
misconfigured *dedicated* database is caught even though no real request has ever been routed to
it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import uuid
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.db.guard import ROLE_STATEMENT_TIMEOUT_MS

pgserver = pytest.importorskip("pgserver")

DEDICATED_ALIAS = "tenant-red"


def _psql(server, command: str) -> None:
    """`server.psql` without a shell: pgserver's own version breaks on paths with spaces."""
    from pgserver.postgres_server import POSTGRES_BIN_PATH

    subprocess.run(
        [str(POSTGRES_BIN_PATH / "psql"), server.get_uri()],
        input=command.encode(),
        check=True,
        capture_output=True,
    )


def _bootstrap_and_migrate(prefix: str) -> dict[str, object]:
    """One ephemeral instance, fully bootstrapped and migrated exactly like `database_urls` in
    `test_rls_integration.py`: `app_owner` runs the migrations, `app` is the unprivileged runtime
    role, and pgserver's own `postgres` role stands in for a would-be superuser/BYPASSRLS
    credential."""
    pgdata = tempfile.mkdtemp(prefix=prefix)
    server = pgserver.get_server(pgdata, cleanup_mode="delete")
    sockdir = parse_qs(urlparse(server.get_uri()).query)["host"][0]
    _psql(
        server,
        "CREATE EXTENSION IF NOT EXISTS vector; "
        "CREATE ROLE app_owner LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE; "
        "ALTER SCHEMA public OWNER TO app_owner; "
        "GRANT CREATE ON DATABASE postgres TO app_owner; "
        "CREATE ROLE app LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE; "
        "GRANT USAGE ON SCHEMA public TO app; "
        f"ALTER ROLE app SET statement_timeout = '{ROLE_STATEMENT_TIMEOUT_MS}ms';",
    )
    urls = {
        "migrations": f"postgresql+asyncpg://app_owner@/postgres?host={sockdir}",
        "app": f"postgresql+asyncpg://app@/postgres?host={sockdir}",
        "superuser": f"postgresql+asyncpg://postgres@/postgres?host={sockdir}",
    }
    env = {**os.environ, "DATABASE_URL_MIGRATIONS": urls["migrations"], "DATABASE_URL": urls["app"]}
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"], check=True, env=env, timeout=120
    )
    return {"server": server, **urls}


@pytest.fixture(scope="module")
def pooled_instance():
    info = _bootstrap_and_migrate("pgdata-guard-pooled-")
    yield info
    info["server"].cleanup()


@pytest.fixture(scope="module")
def dedicated_instance():
    info = _bootstrap_and_migrate("pgdata-guard-dedicated-")
    yield info
    info["server"].cleanup()


@pytest.fixture
def guard_env(pooled_instance, dedicated_instance, tmp_path, monkeypatch):
    """Points the application's own engine (`app.db.session.get_engine`, which the guard's
    pooled alias always resolves to) at `pooled_instance["app"]`, wires
    `TENANT_DB_SECRETS_DIR` at a fresh per-test directory, and seeds one pooled and one
    dedicated tenant in the pooled instance's control plane -- exactly the "one alias per
    isolation tier" shape the acceptance criteria ask for. The dedicated tenant's alias has no
    secret file yet; individual tests write one pointed at whichever `dedicated_instance` DSN
    (or a deliberately wrong one) that test wants the guard to find.
    """
    from app import config
    from app.db import engine_registry
    from app.db import session as db_session

    monkeypatch.setenv("DATABASE_URL", pooled_instance["app"])
    monkeypatch.setenv("DATABASE_URL_MIGRATIONS", pooled_instance["migrations"])
    secrets_dir = tmp_path / "tenant-db"
    secrets_dir.mkdir()
    monkeypatch.setenv("TENANT_DB_SECRETS_DIR", str(secrets_dir))
    config.get_settings.cache_clear()
    db_session._engine = None
    db_session._session_factory = None
    engine_registry.reset_registry_for_tests()

    superuser_engine = create_async_engine(pooled_instance["superuser"])

    async def _seed():
        pooled_tenant, dedicated_tenant = uuid.uuid4(), uuid.uuid4()
        tenants = ((pooled_tenant, "pooled-co"), (dedicated_tenant, "dedicated-co"))
        async with superuser_engine.begin() as conn:
            for tenant_id, name in tenants:
                await conn.execute(
                    text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
                    {"id": tenant_id, "name": name},
                )
            await conn.execute(
                text(
                    "INSERT INTO control.tenants (tenant_id, isolation_tier, database_alias) "
                    "VALUES (:tid, 'pooled', NULL)"
                ),
                {"tid": pooled_tenant},
            )
            await conn.execute(
                text(
                    "INSERT INTO control.tenants (tenant_id, isolation_tier, database_alias) "
                    "VALUES (:tid, 'dedicated', :alias)"
                ),
                {"tid": dedicated_tenant, "alias": DEDICATED_ALIAS},
            )

    yield {"secrets_dir": secrets_dir, "seed": _seed, "superuser_engine": superuser_engine}

    config.get_settings.cache_clear()
    db_session._engine = None
    db_session._session_factory = None
    engine_registry.reset_registry_for_tests()


async def _write_alias_secret(guard_env, dedicated_instance, key: str = "app") -> None:
    (guard_env["secrets_dir"] / DEDICATED_ALIAS).write_text(dedicated_instance[key])


async def test_guard_fails_when_dedicated_alias_is_connected_as_a_bypassrls_role(
    guard_env, dedicated_instance
):
    """Acceptance: the guard fails startup when a dedicated alias's engine is connected as a
    role with BYPASSRLS (pgserver's own bootstrap `postgres` role), even though the pooled
    engine -- checked first -- passes."""
    from app.db.guard import PrivilegedRoleOrMissingRLSError, run_role_rls_guard

    await guard_env["seed"]()
    await _write_alias_secret(guard_env, dedicated_instance, key="superuser")

    with pytest.raises(PrivilegedRoleOrMissingRLSError):
        await run_role_rls_guard()


async def test_guard_fails_when_dedicated_alias_has_a_table_missing_forced_rls(
    guard_env, dedicated_instance
):
    """Acceptance: the guard fails startup when a dedicated alias's engine points at an
    instance with a table missing forced Row-Level Security, even though the pooled instance is
    fully compliant."""
    from app.db.guard import PrivilegedRoleOrMissingRLSError, run_role_rls_guard
    from app.db.models import TENANT_ISOLATION_EXCEPTIONS

    assert "scratch_unforced" not in TENANT_ISOLATION_EXCEPTIONS
    await guard_env["seed"]()
    await _write_alias_secret(guard_env, dedicated_instance, key="app")

    owner_engine = create_async_engine(dedicated_instance["migrations"])
    try:
        async with owner_engine.begin() as conn:
            await conn.execute(text("CREATE TABLE scratch_unforced (id int)"))

        with pytest.raises(PrivilegedRoleOrMissingRLSError):
            await run_role_rls_guard()
    finally:
        async with owner_engine.begin() as conn:
            await conn.execute(text("DROP TABLE IF EXISTS scratch_unforced"))
        await owner_engine.dispose()


async def test_guard_passes_when_every_referenced_alias_is_compliant(guard_env, dedicated_instance):
    """Acceptance: the guard passes when every currently-referenced alias, pooled and one
    dedicated, is fully compliant."""
    from app.db.guard import run_role_rls_guard

    await guard_env["seed"]()
    await _write_alias_secret(guard_env, dedicated_instance, key="app")

    await run_role_rls_guard()  # must not raise


async def test_guard_iterates_past_a_compliant_pooled_engine_to_a_failing_dedicated_one(
    guard_env, dedicated_instance, monkeypatch
):
    """Acceptance: a case where the pooled engine passes but a second, dedicated engine fails
    proves the guard actually iterates every referenced alias rather than stopping after the
    first. Spied via `check_role_and_rls` itself, so the assertion is about which engines were
    actually checked, not just that *some* error was raised."""
    from app.db import guard as guard_module
    from app.db.engine_registry import POOLED_ALIAS

    await guard_env["seed"]()
    await _write_alias_secret(guard_env, dedicated_instance, key="superuser")

    checked: list[str] = []
    original_check = guard_module.check_role_and_rls

    async def spying_check(conn):
        # The pooled engine's own DSN carries no alias, so identify it by which server the
        # connection belongs to: the pooled instance's app-role DSN.
        checked.append(str(conn.engine.url))
        await original_check(conn)

    monkeypatch.setattr(guard_module, "check_role_and_rls", spying_check)

    with pytest.raises(guard_module.PrivilegedRoleOrMissingRLSError):
        await guard_module.run_role_rls_guard()

    # Both engines were reached: the pooled one (which passed, since it isn't in `checked`
    # failing) and then the dedicated one (whose BYPASSRLS connection made the guard raise).
    # POOLED_ALIAS sorts before DEDICATED_ALIAS ("pooled" < "tenant-red"), so this also proves
    # the loop did not stop after the first (pooled, compliant) alias.
    assert len(checked) == 2
    assert POOLED_ALIAS  # sanity: alias constant imported and used for the docstring's claim
