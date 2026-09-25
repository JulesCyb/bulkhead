"""Embedded-Postgres integration test for the operator tool skeleton (Spec 9 / #68): audited
command dispatch, the tenant-lookup helper, and the read-only tenant listing. Pattern:
`tests/test_rls_integration.py`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import ROLE_STATEMENT_TIMEOUT_MS

pgserver = pytest.importorskip("pgserver")


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
def database_urls():
    """Same bootstrap as `tests/test_rls_integration.py`: `app_owner` runs the migrations, the
    real superuser is used only to seed fixtures without a tenant context."""
    pgdata = tempfile.mkdtemp(prefix="pgdata-operator-")
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


async def _seed_tenant(
    superuser_url: str,
    *,
    name: str,
    residency: str = "eu",
    isolation_tier: str = "pooled",
    database_alias: str | None = None,
    suspended: bool = False,
) -> uuid.UUID:
    """One tenant across `public.tenants`, `control.tenants`, and `control.tenant_directory` --
    as the superuser, which bypasses RLS entirely, exactly like `_seed` in
    `tests/test_rls_integration.py`. Standing in for what a future `create` command will do."""
    tenant_id = uuid.uuid4()
    engine = create_async_engine(superuser_url)
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name, settings) VALUES (:id, :name, CAST(:s AS jsonb))"),
            {"id": tenant_id, "name": name, "s": json.dumps({"residency": residency})},
        )
        await conn.execute(
            text(
                "INSERT INTO control.tenants (tenant_id, isolation_tier, database_alias) "
                "VALUES (:id, :tier, :alias)"
            ),
            {"id": tenant_id, "tier": isolation_tier, "alias": database_alias},
        )
        if suspended:
            await conn.execute(
                text("UPDATE control.tenants SET suspended_at = now() WHERE tenant_id = :id"),
                {"id": tenant_id},
            )
        await conn.execute(
            text("INSERT INTO control.tenant_directory (tenant_id, name) VALUES (:id, :name)"),
            {"id": tenant_id, "name": name},
        )
    await engine.dispose()
    return tenant_id


@pytest.fixture
def operator_env(database_urls, monkeypatch):
    """Points the operator tool's settings object at the embedded database, mirroring
    `app_settings` in `tests/test_rls_integration.py`."""
    from app import migration_settings

    monkeypatch.setenv("DATABASE_URL_MIGRATIONS", database_urls["migrations"])
    migration_settings.get_migration_settings.cache_clear()
    yield
    migration_settings.get_migration_settings.cache_clear()


async def test_listing_reports_every_seeded_tenant(database_urls):
    from app.operator.listing import list_tenants

    tenant_a = await _seed_tenant(
        database_urls["superuser"],
        name="Acme",
        residency="eu",
        isolation_tier="pooled",
    )
    tenant_b = await _seed_tenant(
        database_urls["superuser"],
        name="Globex",
        residency="us",
        isolation_tier="dedicated",
        database_alias="globex_db",
        suspended=True,
    )

    engine = create_async_engine(database_urls["migrations"])
    try:
        async with engine.begin() as conn:
            summaries = await list_tenants(conn)
    finally:
        await engine.dispose()

    by_id = {s.tenant_id: s for s in summaries}
    assert tenant_a in by_id and tenant_b in by_id

    acme = by_id[tenant_a]
    assert acme.isolation_tier == "pooled"
    assert acme.residency == "eu"
    assert acme.database_alias is None
    assert acme.suspended is False

    globex = by_id[tenant_b]
    assert globex.isolation_tier == "dedicated"
    assert globex.residency == "us"
    assert globex.database_alias == "globex_db"
    assert globex.suspended is True
    assert globex.suspended_at is not None


async def test_lookup_resolves_by_id_and_by_unambiguous_name(database_urls):
    from app.operator.lookup import resolve_tenant

    tenant_id = await _seed_tenant(database_urls["superuser"], name="Initech")

    engine = create_async_engine(database_urls["migrations"])
    try:
        async with engine.begin() as conn:
            by_id = await resolve_tenant(conn, str(tenant_id))
            by_name = await resolve_tenant(conn, "Initech")
    finally:
        await engine.dispose()

    assert by_id.tenant_id == tenant_id
    assert by_name.tenant_id == tenant_id


async def test_lookup_rejects_ambiguous_name(database_urls):
    from app.operator.lookup import AmbiguousTenantNameError, resolve_tenant

    await _seed_tenant(database_urls["superuser"], name="Dup Co")
    await _seed_tenant(database_urls["superuser"], name="Dup Co")

    engine = create_async_engine(database_urls["migrations"])
    try:
        async with engine.begin() as conn:
            with pytest.raises(AmbiguousTenantNameError):
                await resolve_tenant(conn, "Dup Co")
    finally:
        await engine.dispose()


async def test_lookup_raises_not_found_for_unknown_id_and_name(database_urls):
    from app.operator.lookup import TenantNotFoundError, resolve_tenant

    engine = create_async_engine(database_urls["migrations"])
    try:
        async with engine.begin() as conn:
            with pytest.raises(TenantNotFoundError):
                await resolve_tenant(conn, str(uuid.uuid4()))
            with pytest.raises(TenantNotFoundError):
                await resolve_tenant(conn, "no such tenant")
    finally:
        await engine.dispose()


async def _operator_actions(superuser_url: str, action: str) -> list:
    """Only the superuser can read `control.operator_actions` in this test -- `app_owner`
    itself holds INSERT-only on it by grant (migration 0004), exactly as production requires."""
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
    # asyncpg (via SQLAlchemy's raw text()) may hand back a jsonb column as a decoded dict or
    # as its raw string encoding depending on the driver's codec setup -- normalize either way.
    return [
        (
            tenant_id,
            action,
            json.loads(details) if isinstance(details, str) else details,
            performed_at,
        )
        for tenant_id, action, details, performed_at in rows
    ]


async def test_cli_list_records_the_invocation_in_the_operator_action_log(
    database_urls, operator_env
):
    from app.operator.cli import _run, build_parser

    await _seed_tenant(database_urls["superuser"], name="Umbrella")

    # `_run` directly, not `main()`: `main()` wraps this in `asyncio.run`, which cannot be
    # called from the event loop pytest-asyncio is already running this test in.
    args = build_parser().parse_args(["list"])
    exit_code = await _run(args.command, args)
    assert exit_code == 0

    rows = await _operator_actions(database_urls["superuser"], "list")
    assert len(rows) >= 1
    tenant_id, action, details, performed_at = rows[-1]
    # `list` targets every tenant, not one -- the documented nil-UUID sentinel, never NULL
    # (control.operator_actions.tenant_id is NOT NULL).
    assert str(tenant_id) == "00000000-0000-0000-0000-000000000000"
    assert action == "list"
    assert details["outcome"].startswith("ok: listed ")
    assert "operator" in details
    assert "started_at" in details and "finished_at" in details
    assert isinstance(details["duration_ms"], int)
    assert performed_at is not None


async def test_cli_records_a_failed_invocation_too(database_urls, operator_env, monkeypatch):
    """Even a failed command is recorded, with its error, not silently dropped."""
    import app.operator.cli as cli_module

    async def _boom(conn, args):
        raise RuntimeError("simulated failure")

    monkeypatch.setitem(cli_module._COMMANDS, "list", _boom)

    args = cli_module.build_parser().parse_args(["list"])
    exit_code = await cli_module._run(args.command, args)
    assert exit_code == 1

    rows = await _operator_actions(database_urls["superuser"], "list")
    _, _, details, _ = rows[-1]
    assert details["outcome"] == "error"
    assert "simulated failure" in details["error"]


def test_cli_has_no_flag_or_env_var_for_an_alternate_connection_string():
    """Acceptance: the tool connects only as the owner role's DSN; nothing accepts or falls back
    to a superuser (or any other) connection string."""
    from app.operator.cli import build_parser

    parser = build_parser()
    option_strings = {
        opt for action in parser._actions for opt in getattr(action, "option_strings", [])
    }
    assert not any("dsn" in opt.lower() or "database" in opt.lower() for opt in option_strings)


def test_cli_module_never_imports_the_application_settings_object():
    """`app.config.Settings` is the long-running API's settings object; the operator tool must
    only ever read `app.migration_settings` (the owner DSN)."""
    import ast

    import app.operator.cli as cli_module

    with open(cli_module.__file__) as f:
        tree = ast.parse(f.read())

    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert "app.config" not in imported_modules
    assert imported_modules & {"app.migration_settings"}
