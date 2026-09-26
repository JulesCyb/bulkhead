"""Embedded-Postgres integration tests for the dedicated-tenant path of the `create` command
(#71, ADR-0002, ADR-0010, ADR-0011): actually provisioning a second physical database, applying
the current schema to it, and recording its alias in the control plane -- all in the same
invocation, idempotently. Two ephemeral `pgserver` instances stand in for the pooled
control-plane database and the target server a dedicated tenant's own database is provisioned on
(mirroring `tests/test_tenant_session_routing_integration.py`'s own two-instance pattern).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.db.guard import ROLE_STATEMENT_TIMEOUT_MS
from app.gateway_provisioning import GatewayAdminClient

pgserver = pytest.importorskip("pgserver")

import scripts.migrate as migrate_module  # noqa: E402


def _psql(server, command: str) -> None:
    """`server.psql` without a shell: pgserver's own version breaks on paths with spaces."""
    from pgserver.postgres_server import POSTGRES_BIN_PATH

    subprocess.run(
        [str(POSTGRES_BIN_PATH / "psql"), server.get_uri()],
        input=command.encode(),
        check=True,
        capture_output=True,
    )


@pytest.fixture(scope="module")
def pooled_urls():
    """The pooled control-plane database, fully migrated, `app_owner` running the migrations --
    same bootstrap as `tests/test_operator_tool_integration.py`."""
    pgdata = tempfile.mkdtemp(prefix="pgdata-operator-dedicated-pooled-")
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
    yield urls
    server.cleanup()


@pytest.fixture
def dedicated_target():
    """A bare Postgres server with only the bootstrap superuser and no database but the default
    maintenance one -- standing in for a managed Postgres server `create` provisions a dedicated
    tenant's brand-new database onto (mirrors `scripts/provision_roles.py`'s own "no first-boot
    init hook" scenario, one step earlier: here the database itself does not exist yet either)."""
    pgdata = tempfile.mkdtemp(prefix="pgdata-operator-dedicated-target-")
    server = pgserver.get_server(pgdata, cleanup_mode="delete")
    sockdir = parse_qs(urlparse(server.get_uri()).query)["host"][0]
    admin_url = f"postgresql+asyncpg://postgres@/postgres?host={sockdir}"
    yield {"sockdir": sockdir, "admin_url": admin_url}
    server.cleanup()


@pytest.fixture
def operator_env(pooled_urls, monkeypatch):
    from app import migration_settings

    monkeypatch.setenv("DATABASE_URL_MIGRATIONS", pooled_urls["migrations"])
    migration_settings.get_migration_settings.cache_clear()
    yield
    migration_settings.get_migration_settings.cache_clear()


@pytest.fixture
def dedicated_secrets_dirs(tmp_path, monkeypatch):
    """Points scripts/migrate.py and app/db/engine_registry.py's secret-file directories at
    fresh, empty tmp directories, exactly as a real deployment would provide separately."""
    migrations_dir = tmp_path / "tenant-db-migrations"
    migrations_dir.mkdir()
    app_dir = tmp_path / "tenant-db"
    app_dir.mkdir()
    monkeypatch.setenv("TENANT_DB_MIGRATIONS_SECRETS_DIR", str(migrations_dir))
    monkeypatch.setenv("TENANT_DB_SECRETS_DIR", str(app_dir))
    return {"migrations": migrations_dir, "app": app_dir}


def _fake_admin_client(*, key: str = "sk-minted") -> GatewayAdminClient:
    """Mirrors `tests/test_operator_tool_integration.py`'s own fake -- the gateway is never
    reached over a real network here either."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/key/generate":
            return httpx.Response(200, json={"key": key})
        return httpx.Response(404)

    http_client = httpx.AsyncClient(
        base_url="http://litellm.internal:4000", transport=httpx.MockTransport(handler)
    )
    return GatewayAdminClient(
        base_url="http://litellm.internal:4000",
        master_key="sk-master-test",
        http_client=http_client,
    )


def _owner_dsn(alias: str, sockdir: str) -> str:
    return f"postgresql+asyncpg://app_owner@/{alias}?host={sockdir}"


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


def _head_revision() -> str:
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    config = Config(str(migrate_module._ALEMBIC_INI))
    config.set_main_option("script_location", str(migrate_module._REPO_ROOT / "migrations"))
    return ScriptDirectory.from_config(config).get_current_head()


async def test_create_provisions_a_second_physical_database_at_head(
    pooled_urls, operator_env, dedicated_target, dedicated_secrets_dirs, tmp_path
):
    """Acceptance (#71): creating a dedicated tenant results in a second physical database that
    exists, carries the current migration head, and is named by the alias the control-plane
    record stores; its admin membership lives there too, never in the pooled database."""
    from app.config import Settings
    from app.operator.create import create_tenant

    settings = Settings(gateway_credentials_dir=str(tmp_path))
    engine = create_async_engine(pooled_urls["migrations"])
    try:
        async with engine.begin() as conn:
            result = await create_tenant(
                conn,
                tenant_name="Dedicated Co",
                residency="eu",
                admin_email="admin@dedicated.test",
                isolation_tier="dedicated",
                dedicated_db_admin_url=dedicated_target["admin_url"],
                settings=settings,
                admin_client=_fake_admin_client(key="sk-dedicated-co"),
            )
    finally:
        await engine.dispose()

    assert result.isolation_tier == "dedicated"
    assert result.database_alias is not None
    assert result.dedicated_database == "provisioned"
    assert result.admin_membership == "created"
    assert result.control_plane == "created"

    owner_dsn = _owner_dsn(result.database_alias, dedicated_target["sockdir"])
    version = await _alembic_version(owner_dsn)
    assert version == _head_revision()

    # The control-plane record (in the pooled database) carries the alias.
    verify_engine = create_async_engine(pooled_urls["superuser"])
    try:
        async with verify_engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text(
                            "SELECT isolation_tier, database_alias FROM control.tenants "
                            "WHERE tenant_id = :tid"
                        ),
                        {"tid": result.tenant_id},
                    )
                )
                .mappings()
                .one()
            )
    finally:
        await verify_engine.dispose()
    assert row["isolation_tier"] == "dedicated"
    assert row["database_alias"] == result.database_alias

    # The membership lives in the dedicated database...
    dedicated_engine = create_async_engine(owner_dsn)
    try:
        async with dedicated_engine.begin() as conn:
            await conn.execute(
                text("SELECT set_config('app.tenant_id', :tid, true)"),
                {"tid": str(result.tenant_id)},
            )
            membership = (
                await conn.execute(
                    text(
                        "SELECT role FROM memberships WHERE tenant_id = :tid AND identity_id = :iid"
                    ),
                    {"tid": result.tenant_id, "iid": result.identity_id},
                )
            ).one()
    finally:
        await dedicated_engine.dispose()
    assert membership.role == "admin"

    # ...never in the pooled one.
    pooled_membership_engine = create_async_engine(pooled_urls["superuser"])
    try:
        async with pooled_membership_engine.connect() as conn:
            count = (
                await conn.execute(
                    text("SELECT count(*) FROM memberships WHERE tenant_id = :tid"),
                    {"tid": result.tenant_id},
                )
            ).scalar_one()
    finally:
        await pooled_membership_engine.dispose()
    assert count == 0

    # Both tenant-secret files were written.
    assert (dedicated_secrets_dirs["migrations"] / result.database_alias).exists()
    assert (dedicated_secrets_dirs["app"] / result.database_alias).exists()


async def test_rerun_does_not_recreate_the_database_or_reapply_migrations(
    pooled_urls, operator_env, dedicated_target, dedicated_secrets_dirs, tmp_path
):
    """Acceptance (#71): re-running create against the same dedicated tenant does not attempt to
    recreate the database or reapply migrations it already applied -- proven by omitting
    `--dedicated-db-admin-url` entirely on the second call (it would be required if any
    provisioning were attempted again) and by a second, differently-keyed fake gateway client
    that must never actually be minted from."""
    from app.config import Settings
    from app.operator.create import create_tenant

    settings = Settings(gateway_credentials_dir=str(tmp_path))
    engine = create_async_engine(pooled_urls["migrations"])
    try:
        async with engine.begin() as conn:
            first = await create_tenant(
                conn,
                tenant_name="Rerun Dedicated Co",
                residency="eu",
                admin_email="admin@rerun-dedicated.test",
                isolation_tier="dedicated",
                dedicated_db_admin_url=dedicated_target["admin_url"],
                settings=settings,
                admin_client=_fake_admin_client(key="sk-rerun-dedicated"),
            )
    finally:
        await engine.dispose()

    second_engine = create_async_engine(pooled_urls["migrations"])
    try:
        async with second_engine.begin() as conn:
            second = await create_tenant(
                conn,
                tenant_name="Rerun Dedicated Co",
                residency="eu",
                admin_email="admin@rerun-dedicated.test",
                isolation_tier="dedicated",
                dedicated_db_admin_url=None,  # never needed again -- see docstring
                settings=settings,
                admin_client=_fake_admin_client(key="sk-should-not-be-minted"),
            )
    finally:
        await second_engine.dispose()

    assert second.tenant_id == first.tenant_id
    assert second.database_alias == first.database_alias
    assert second.control_plane == "already exists"
    assert second.dedicated_database == "already provisioned"
    assert second.admin_membership == "already exists"
    assert second.gateway_credential == "already provisioned"
    assert second.gateway_credential_alias == first.gateway_credential_alias


async def test_pooled_tenant_never_gets_a_database_alias(pooled_urls, operator_env, tmp_path):
    """Acceptance (#71): the control-plane record's database-alias field is populated only for
    the dedicated tier, and left absent for a pooled tenant."""
    from app.config import Settings
    from app.operator.create import create_tenant

    settings = Settings(gateway_credentials_dir=str(tmp_path))
    engine = create_async_engine(pooled_urls["migrations"])
    try:
        async with engine.begin() as conn:
            result = await create_tenant(
                conn,
                tenant_name="Still Pooled Co",
                residency="eu",
                admin_email="admin@still-pooled.test",
                settings=settings,
                admin_client=_fake_admin_client(key="sk-still-pooled"),
            )
    finally:
        await engine.dispose()

    assert result.isolation_tier == "pooled"
    assert result.database_alias is None
    assert result.dedicated_database is None


async def test_create_dedicated_is_recorded_in_the_operator_action_log(
    pooled_urls, operator_env, dedicated_target, dedicated_secrets_dirs, tmp_path, monkeypatch
):
    """Acceptance (#71): the create invocation for a dedicated tenant is recorded in the
    operator-action log the same way a pooled create is, with the admin URL redacted."""
    import app.operator.create as create_module
    from app import config
    from app.operator.cli import _run, build_parser

    monkeypatch.setenv("GATEWAY_CREDENTIALS_DIR", str(tmp_path))
    config.get_settings.cache_clear()
    monkeypatch.setattr(
        create_module,
        "build_admin_client",
        lambda settings: _fake_admin_client(key="sk-cli-dedicated"),
    )

    args = build_parser().parse_args(
        [
            "create",
            "CLI Dedicated Co",
            "--residency",
            "eu",
            "--admin-email",
            "admin@cli-dedicated.test",
            "--isolation-tier",
            "dedicated",
            "--dedicated-db-admin-url",
            dedicated_target["admin_url"],
        ]
    )
    exit_code = await _run(args.command, args)
    config.get_settings.cache_clear()
    assert exit_code == 0

    rows_engine = create_async_engine(pooled_urls["superuser"])
    try:
        async with rows_engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT tenant_id, action, details FROM control.operator_actions "
                        "WHERE action = 'create' ORDER BY performed_at"
                    )
                )
            ).all()
    finally:
        await rows_engine.dispose()

    assert len(rows) >= 1
    _, action, details = rows[-1]
    if isinstance(details, str):
        details = json.loads(details)
    assert action == "create"
    assert details["outcome"].startswith("ok: tenant ")
    assert details["args"]["isolation_tier"] == "dedicated"
    assert details["args"]["dedicated_db_admin_url"] == "<redacted>"
