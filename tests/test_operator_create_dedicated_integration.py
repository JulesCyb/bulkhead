"""Embedded-Postgres integration tests for the dedicated-tenant path of the `create` command
(#71, ADR-0002, ADR-0010, ADR-0011): actually provisioning a second physical database, applying
the current schema to it, and recording its alias in the control plane -- all in the same
invocation, idempotently.

The pooled control-plane database comes from `tests.support` (issue #96 / spec #90, "A6"); this
file's own private bootstrap of it was deleted in favor of that shared fixture. `dedicated_target`
below -- a second, bare `pgserver` instance standing in for the separate managed-Postgres server a
dedicated tenant's own database is provisioned onto -- stays private to this file (and to
`tests/test_operator_erase_integration.py`, which mirrors it under the same name): it is not
seeding, it is the one thing these tests exist to prove (`create`/`erase` really reach a *second*
server via `--dedicated-db-admin-url`), so `tests.support.seeding.seed_tenant`'s own dedicated
path deliberately does not need it (it passes the *same* cluster's own superuser URL instead --
see that module's docstring) and neither should this file's `pooled_urls` cluster.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

pgserver = pytest.importorskip("pgserver")

import scripts.migrate as migrate_module  # noqa: E402
from tests.support import cluster, environment  # noqa: E402
from tests.support.gateway import fake_gateway_admin_client  # noqa: E402

# `cluster`/`environment` are imported only so pytest can discover them as fixtures from this
# module's namespace -- referenced only by parameter name in the tests below, never called
# directly.
_ = (cluster, environment)


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
    environment, dedicated_target, tmp_path
):
    """Acceptance (#71): creating a dedicated tenant results in a second physical database that
    exists, carries the current migration head, and is named by the alias the control-plane
    record stores; its admin membership lives there too, never in the pooled database."""
    from app.config import Settings
    from app.operator.create import create_tenant

    settings = Settings(gateway_credentials_dir=str(tmp_path))
    engine = create_async_engine(environment.owner_url)
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
                admin_client=fake_gateway_admin_client(key="sk-dedicated-co"),
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
    verify_engine = create_async_engine(environment.superuser_url)
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
    pooled_membership_engine = create_async_engine(environment.superuser_url)
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
    assert (Path(os.environ["TENANT_DB_MIGRATIONS_SECRETS_DIR"]) / result.database_alias).exists()
    assert (Path(os.environ["TENANT_DB_SECRETS_DIR"]) / result.database_alias).exists()


async def test_rerun_does_not_recreate_the_database_or_reapply_migrations(
    environment, dedicated_target, tmp_path
):
    """Acceptance (#71): re-running create against the same dedicated tenant does not attempt to
    recreate the database or reapply migrations it already applied -- proven by omitting
    `--dedicated-db-admin-url` entirely on the second call (it would be required if any
    provisioning were attempted again) and by a second, differently-keyed fake gateway client
    that must never actually be minted from."""
    from app.config import Settings
    from app.operator.create import create_tenant

    settings = Settings(gateway_credentials_dir=str(tmp_path))
    engine = create_async_engine(environment.owner_url)
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
                admin_client=fake_gateway_admin_client(key="sk-rerun-dedicated"),
            )
    finally:
        await engine.dispose()

    second_engine = create_async_engine(environment.owner_url)
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
                admin_client=fake_gateway_admin_client(key="sk-should-not-be-minted"),
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


async def test_pooled_tenant_never_gets_a_database_alias(environment, tmp_path):
    """Acceptance (#71): the control-plane record's database-alias field is populated only for
    the dedicated tier, and left absent for a pooled tenant."""
    from app.config import Settings
    from app.operator.create import create_tenant

    settings = Settings(gateway_credentials_dir=str(tmp_path))
    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            result = await create_tenant(
                conn,
                tenant_name="Still Pooled Co",
                residency="eu",
                admin_email="admin@still-pooled.test",
                settings=settings,
                admin_client=fake_gateway_admin_client(key="sk-still-pooled"),
            )
    finally:
        await engine.dispose()

    assert result.isolation_tier == "pooled"
    assert result.database_alias is None
    assert result.dedicated_database is None


async def test_create_dedicated_is_recorded_in_the_operator_action_log(
    environment, dedicated_target, tmp_path, monkeypatch
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
        lambda settings: fake_gateway_admin_client(key="sk-cli-dedicated"),
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

    rows_engine = create_async_engine(environment.superuser_url)
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
