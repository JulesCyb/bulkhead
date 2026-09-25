"""Embedded-Postgres integration test for the operator tool skeleton (Spec 9 / #68): audited
command dispatch, the tenant-lookup helper, and the read-only tenant listing. Pattern:
`tests/test_rls_integration.py`.

Also covers the `create` command (Spec 9 / #70): provisioning a pooled tenant end to end, its
idempotency, up-front residency/model validation, and the audit log. The gateway is never
reached over a real network here either -- `GatewayAdminClient` is always built on
`httpx.MockTransport`, the same pattern `tests/test_gateway_provisioning_integration.py` uses.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import ROLE_STATEMENT_TIMEOUT_MS
from app.gateway_provisioning import GatewayAdminClient

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
    """One tenant across `public.tenants` and `control.tenants` -- as the superuser, which
    bypasses RLS entirely, exactly like `_seed` in `tests/test_rls_integration.py`. Standing in
    for what a future `create` command will do."""
    tenant_id = uuid.uuid4()
    engine = create_async_engine(superuser_url)
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant_id, "name": name},
        )
        await conn.execute(
            text(
                "INSERT INTO control.tenants "
                "(tenant_id, isolation_tier, database_alias, residency) "
                "VALUES (:id, :tier, :alias, :residency)"
            ),
            {
                "id": tenant_id,
                "tier": isolation_tier,
                "alias": database_alias,
                "residency": residency,
            },
        )
        if suspended:
            await conn.execute(
                text("UPDATE control.tenants SET suspended_at = now() WHERE tenant_id = :id"),
                {"id": tenant_id},
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


async def test_app_cannot_widen_its_view_with_the_operator_read_flag(database_urls):
    """Regression (mirrors `test_app_cannot_widen_its_control_plane_view_with_the_migration_read_
    flag` in `tests/test_rls_integration.py` for 0016's identically-shaped flag): `app` can set
    any custom setting in its own session, and a real `app` session querying
    `control.tenants_view`/`tenants` still has `current_user = 'app'` (views check table
    permissions as their owner but do not reassign `current_user`, unlike a `SECURITY DEFINER`
    function). So `control_tenants_operator_read`/`tenants_operator_read` requiring `current_user
    = 'app_owner'`, not just the flag, must still leave `app` seeing only its own tenant even
    after setting `app.control_operator_read` itself."""
    tenant_a = await _seed_tenant(database_urls["superuser"], name="Flag A")
    await _seed_tenant(database_urls["superuser"], name="Flag B")

    engine = create_async_engine(database_urls["app"])
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT set_config('app.control_operator_read', 'true', true)"))
            await conn.execute(
                text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": str(tenant_a)}
            )
            visible = (
                (await conn.execute(text("SELECT tenant_id FROM control.tenants_view")))
                .scalars()
                .all()
            )
            visible_public = (await conn.execute(text("SELECT id FROM tenants"))).scalars().all()
    finally:
        await engine.dispose()
    assert visible == [tenant_a]
    assert visible_public == [tenant_a]


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


async def test_suspend_sets_the_flag_and_timestamp_and_reruns_as_a_no_op(database_urls):
    """Spec 9 / #69, ADR-0010, seam 1: `suspend` sets `suspended`/`suspended_at`, and running it
    again against an already-suspended tenant reports a no-op, not an error."""
    from app.operator.suspend import set_tenant_suspended

    tenant_id = await _seed_tenant(database_urls["superuser"], name="Suspend Co")

    engine = create_async_engine(database_urls["migrations"])
    try:
        async with engine.begin() as conn:
            first = await set_tenant_suspended(conn, str(tenant_id), suspended=True)
        async with engine.begin() as conn:
            second = await set_tenant_suspended(conn, str(tenant_id), suspended=True)
    finally:
        await engine.dispose()

    assert first.changed is True
    assert first.suspended is True
    assert first.suspended_at is not None

    assert second.changed is False
    assert second.suspended is True
    assert second.suspended_at == first.suspended_at


async def test_unsuspend_restores_the_tenant_with_nothing_reprovisioned(database_urls):
    """`unsuspend` clears `suspended_at` and is itself a no-op when the tenant is already
    active -- and never touches isolation tier, database alias, or residency."""
    from app.operator.listing import list_tenants
    from app.operator.suspend import set_tenant_suspended

    tenant_id = await _seed_tenant(
        database_urls["superuser"],
        name="Restore Co",
        residency="us",
        isolation_tier="dedicated",
        database_alias="restore_db",
        suspended=True,
    )

    engine = create_async_engine(database_urls["migrations"])
    try:
        async with engine.begin() as conn:
            result = await set_tenant_suspended(conn, str(tenant_id), suspended=False)
        async with engine.begin() as conn:
            no_op = await set_tenant_suspended(conn, str(tenant_id), suspended=False)
        async with engine.begin() as conn:
            summaries = {t.tenant_id: t for t in await list_tenants(conn)}
    finally:
        await engine.dispose()

    assert result.changed is True
    assert result.suspended is False
    assert result.suspended_at is None

    assert no_op.changed is False

    restored = summaries[tenant_id]
    assert restored.suspended is False
    assert restored.isolation_tier == "dedicated"
    assert restored.database_alias == "restore_db"
    assert restored.residency == "us"


async def test_suspend_creates_a_control_plane_row_for_a_pooled_tenant_with_none_yet(
    database_urls,
):
    """A tenant provisioned only through the pooled default (ADR-0002, `scripts/seed.py`'s
    original path) has no `control.tenants` row at all -- suspending it must not error, it
    creates one."""
    from app.operator.suspend import set_tenant_suspended

    tenant_id = uuid.uuid4()
    engine = create_async_engine(database_urls["superuser"])
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant_id, "name": "No Control Row Co"},
        )
    await engine.dispose()

    engine = create_async_engine(database_urls["migrations"])
    try:
        async with engine.begin() as conn:
            result = await set_tenant_suspended(conn, str(tenant_id), suspended=True)
    finally:
        await engine.dispose()

    assert result.changed is True
    assert result.suspended is True
    assert result.suspended_at is not None


async def test_app_cannot_widen_its_control_plane_write_with_the_operator_write_flag(
    database_urls,
):
    """Mirrors `test_app_cannot_widen_its_view_with_the_operator_read_flag`: `app` setting
    `app.control_operator_write` itself must not grant it write access to `control.tenants` --
    `current_user = 'app_owner'` is what actually restricts the escape hatch, not the flag alone."""
    tenant_id = await _seed_tenant(database_urls["superuser"], name="Write Flag Co")

    engine = create_async_engine(database_urls["app"])
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text("SELECT set_config('app.control_operator_write', 'true', true)")
            )
            with pytest.raises(Exception):  # noqa: B017 - asyncpg's InsufficientPrivilegeError
                await conn.execute(
                    text("UPDATE control.tenants SET suspended_at = now() WHERE tenant_id = :tid"),
                    {"tid": str(tenant_id)},
                )
    finally:
        await engine.dispose()


async def test_cli_suspend_is_idempotent_and_records_both_invocations_in_the_audit_log(
    database_urls, operator_env
):
    from app.operator.cli import _run, build_parser

    tenant_id = await _seed_tenant(database_urls["superuser"], name="Audited Suspend Co")

    args = build_parser().parse_args(["suspend", str(tenant_id)])
    assert await _run(args.command, args) == 0
    assert await _run(args.command, args) == 0

    rows = await _operator_actions(database_urls["superuser"], "suspend")
    assert len(rows) >= 2
    first_tenant_id, _, first_details, _ = rows[-2]
    second_tenant_id, _, second_details, _ = rows[-1]
    assert str(first_tenant_id) == str(tenant_id)
    assert str(second_tenant_id) == str(tenant_id)
    assert first_details["outcome"].startswith("ok: ")
    assert second_details["outcome"].startswith("no-op: ")


async def test_cli_unsuspend_records_the_invocation_by_tenant_name(database_urls, operator_env):
    from app.operator.cli import _run, build_parser

    tenant_id = await _seed_tenant(
        database_urls["superuser"], name="Named Unsuspend Co", suspended=True
    )

    args = build_parser().parse_args(["unsuspend", "Named Unsuspend Co"])
    exit_code = await _run(args.command, args)
    assert exit_code == 0

    rows = await _operator_actions(database_urls["superuser"], "unsuspend")
    tenant_id_logged, _, details, _ = rows[-1]
    assert str(tenant_id_logged) == str(tenant_id)
    assert details["outcome"].startswith("ok: ")


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


def _fake_admin_client(*, key: str = "sk-minted") -> GatewayAdminClient:
    """Mirrors `tests/test_gateway_provisioning_integration.py`'s own fake -- `create`'s gateway
    call is never reached over a real network here either."""

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


async def test_create_provisions_control_plane_credential_and_admin_membership(
    database_urls, operator_env, tmp_path
):
    """Acceptance (#70): `create` writes the control-plane record (isolation tier, residency,
    database alias), mints and writes a gateway-credential secret file, and creates the first
    admin membership for the named identity -- all in one call."""
    from app.config import Settings
    from app.operator.create import create_tenant

    settings = Settings(gateway_credentials_dir=str(tmp_path))
    engine = create_async_engine(database_urls["migrations"])
    try:
        async with engine.begin() as conn:
            result = await create_tenant(
                conn,
                tenant_name="Create Co",
                residency="eu",
                admin_email="admin@create.test",
                settings=settings,
                admin_client=_fake_admin_client(key="sk-create-co"),
            )
    finally:
        await engine.dispose()

    assert result.control_plane == "created"
    assert result.admin_membership == "created"
    assert result.gateway_credential == "provisioned"
    assert (tmp_path / result.gateway_credential_alias).read_text() == "sk-create-co"

    verify_engine = create_async_engine(database_urls["superuser"])
    try:
        async with verify_engine.connect() as conn:
            tenant_row = (
                (
                    await conn.execute(
                        text(
                            "SELECT residency, isolation_tier, database_alias, "
                            "gateway_credential_alias FROM control.tenants WHERE tenant_id = :tid"
                        ),
                        {"tid": result.tenant_id},
                    )
                )
                .mappings()
                .one()
            )
            membership_row = (
                await conn.execute(
                    text(
                        "SELECT role FROM memberships WHERE tenant_id = :tid AND identity_id = :iid"
                    ),
                    {"tid": result.tenant_id, "iid": result.identity_id},
                )
            ).one()
    finally:
        await verify_engine.dispose()

    assert tenant_row["residency"] == "eu"
    assert tenant_row["isolation_tier"] == "pooled"
    assert tenant_row["database_alias"] is None
    assert tenant_row["gateway_credential_alias"] == result.gateway_credential_alias
    assert membership_row.role == "admin"


async def test_create_is_idempotent_on_rerun(database_urls, operator_env, tmp_path):
    """Acceptance (#70): re-running `create` against the same tenant name performs none of the
    three steps again and reports each as already in place. A second, differently-keyed fake
    admin client proves the gateway is never called a second time -- if it were, the freshly
    minted key would overwrite the alias's file and the final assertion would fail."""
    from app.config import Settings
    from app.operator.create import create_tenant

    settings = Settings(gateway_credentials_dir=str(tmp_path))

    engine = create_async_engine(database_urls["migrations"])
    try:
        async with engine.begin() as conn:
            first = await create_tenant(
                conn,
                tenant_name="Idempotent Co",
                residency="us",
                admin_email="admin@idempotent.test",
                settings=settings,
                admin_client=_fake_admin_client(key="sk-idempotent"),
            )
    finally:
        await engine.dispose()

    assert (first.control_plane, first.admin_membership, first.gateway_credential) == (
        "created",
        "created",
        "provisioned",
    )

    second_engine = create_async_engine(database_urls["migrations"])
    try:
        async with second_engine.begin() as conn:
            second = await create_tenant(
                conn,
                tenant_name="Idempotent Co",
                residency="us",
                admin_email="admin@idempotent.test",
                settings=settings,
                admin_client=_fake_admin_client(key="sk-should-not-be-minted"),
            )
    finally:
        await second_engine.dispose()

    assert second.tenant_id == first.tenant_id
    assert second.identity_id == first.identity_id
    assert (second.control_plane, second.admin_membership, second.gateway_credential) == (
        "already exists",
        "already exists",
        "already provisioned",
    )
    assert second.gateway_credential_alias == first.gateway_credential_alias
    assert (tmp_path / first.gateway_credential_alias).read_text() == "sk-idempotent"


async def test_create_rejects_unrecognized_residency_before_any_write(
    database_urls, operator_env, tmp_path
):
    from app.config import Settings
    from app.operator.create import UnrecognizedResidencyError, create_tenant

    settings = Settings(gateway_credentials_dir=str(tmp_path))
    engine = create_async_engine(database_urls["migrations"])
    try:
        async with engine.begin() as conn:
            with pytest.raises(UnrecognizedResidencyError):
                await create_tenant(
                    conn,
                    tenant_name="Should Not Exist",
                    residency="mars",
                    admin_email="nobody@example.test",
                    settings=settings,
                    admin_client=_fake_admin_client(),
                )
    finally:
        await engine.dispose()

    verify_engine = create_async_engine(database_urls["superuser"])
    try:
        async with verify_engine.connect() as conn:
            count = (
                await conn.execute(
                    text("SELECT count(*) FROM tenants WHERE name = 'Should Not Exist'")
                )
            ).scalar_one()
    finally:
        await verify_engine.dispose()
    assert count == 0


async def test_create_rejects_unrecognized_model_before_any_write(
    database_urls, operator_env, tmp_path
):
    from app.config import Settings
    from app.operator.create import UnrecognizedModelError, create_tenant

    settings = Settings(gateway_credentials_dir=str(tmp_path))
    engine = create_async_engine(database_urls["migrations"])
    try:
        async with engine.begin() as conn:
            with pytest.raises(UnrecognizedModelError):
                await create_tenant(
                    conn,
                    tenant_name="Should Not Exist Either",
                    residency="eu",
                    admin_email="nobody2@example.test",
                    model="gpt-nonexistent",
                    settings=settings,
                    admin_client=_fake_admin_client(),
                )
    finally:
        await engine.dispose()

    verify_engine = create_async_engine(database_urls["superuser"])
    try:
        async with verify_engine.connect() as conn:
            count = (
                await conn.execute(
                    text("SELECT count(*) FROM tenants WHERE name = 'Should Not Exist Either'")
                )
            ).scalar_one()
    finally:
        await verify_engine.dispose()
    assert count == 0


async def test_cli_create_records_the_invocation_in_the_operator_action_log(
    database_urls, operator_env, tmp_path, monkeypatch
):
    """Acceptance (#70): a `create` invocation through the real CLI dispatch is recorded in the
    operator-action log, with secrets redacted from the logged arguments (none of `create`'s own
    arguments are secret-shaped, so this also proves ordinary arguments still show up plainly)."""
    import app.operator.create as create_module
    from app import config
    from app.operator.cli import _run, build_parser

    monkeypatch.setenv("GATEWAY_CREDENTIALS_DIR", str(tmp_path))
    config.get_settings.cache_clear()
    monkeypatch.setattr(
        create_module, "build_admin_client", lambda settings: _fake_admin_client(key="sk-cli")
    )

    args = build_parser().parse_args(
        ["create", "CLI Co", "--residency", "eu", "--admin-email", "admin@cli.test"]
    )
    exit_code = await _run(args.command, args)
    config.get_settings.cache_clear()
    assert exit_code == 0

    rows = await _operator_actions(database_urls["superuser"], "create")
    assert len(rows) >= 1
    _, action, details, performed_at = rows[-1]
    assert action == "create"
    assert details["outcome"].startswith("ok: tenant ")
    assert details["args"]["tenant_name"] == "CLI Co"
    assert details["args"]["admin_email"] == "admin@cli.test"
    assert performed_at is not None
