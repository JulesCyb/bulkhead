"""Embedded-Postgres integration tests for the `erase` command (Spec 9 / #72, ADR-0010): erasing
a suspended tenant everywhere its data lives, recorded, re-runnable, dry-runnable. Pattern:
`tests/test_operator_tool_integration.py` (pooled bootstrap) and
`tests/test_operator_create_dedicated_integration.py` (two-`pgserver`-instance dedicated
bootstrap). The gateway is never reached over a real network -- `GatewayAdminClient` is always
built on `httpx.MockTransport`. Spec 8's tracing backend does not exist yet either: every test
passes its own fake `trace_deleter`, standing in for the per-tenant trace-deletion capability
ADR-0010 says `erase` must call.
"""

from __future__ import annotations

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

from app.config import ROLE_STATEMENT_TIMEOUT_MS, Settings
from app.gateway_provisioning import GatewayAdminClient
from app.operator.create import create_tenant
from app.operator.erase import EraseResult, TenantNotSuspendedError, erase_tenant, record_erasure
from app.operator.suspend import set_tenant_suspended

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
def pooled_urls():
    """The pooled control-plane database, fully migrated -- same bootstrap as
    `tests/test_operator_tool_integration.py`."""
    pgdata = tempfile.mkdtemp(prefix="pgdata-operator-erase-pooled-")
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
    """A bare Postgres server standing in for the server a dedicated tenant's own database is
    provisioned on and later dropped from -- mirrors
    `tests/test_operator_create_dedicated_integration.py`'s own fixture of the same name."""
    pgdata = tempfile.mkdtemp(prefix="pgdata-operator-erase-dedicated-")
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
    migrations_dir = tmp_path / "tenant-db-migrations"
    migrations_dir.mkdir()
    app_dir = tmp_path / "tenant-db"
    app_dir.mkdir()
    monkeypatch.setenv("TENANT_DB_MIGRATIONS_SECRETS_DIR", str(migrations_dir))
    monkeypatch.setenv("TENANT_DB_SECRETS_DIR", str(app_dir))
    return {"migrations": migrations_dir, "app": app_dir}


def _fake_admin_client(
    *, key: str = "sk-minted", fail_deletes_until: int = 0
) -> GatewayAdminClient:
    """A `GatewayAdminClient` on `httpx.MockTransport` handling both `/key/generate` (`create`)
    and `/key/delete` (`erase`'s revoke step) -- never a real network call.

    `fail_deletes_until`: the first this-many `/key/delete` calls return a 500, simulating a
    transient gateway failure for the partial-failure/re-run test; 0 (the default) always
    succeeds. `client.delete_calls` counts every `/key/delete` request made, for assertions.
    """
    state = {"delete_calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/key/generate":
            return httpx.Response(200, json={"key": key})
        if request.url.path == "/key/delete":
            state["delete_calls"] += 1
            if state["delete_calls"] <= fail_deletes_until:
                return httpx.Response(500, text="gateway temporarily unavailable")
            return httpx.Response(200, json={})
        return httpx.Response(404)

    http_client = httpx.AsyncClient(
        base_url="http://litellm.internal:4000", transport=httpx.MockTransport(handler)
    )
    client = GatewayAdminClient(
        base_url="http://litellm.internal:4000",
        master_key="sk-master-test",
        http_client=http_client,
    )
    client.delete_calls = state  # test-only attribute, not part of GatewayAdminClient's contract
    return client


def _fake_trace_deleter():
    calls: list = []

    async def _delete(tenant_id):
        calls.append(tenant_id)

    _delete.calls = calls
    return _delete


async def _membership_id(url: str, *, tenant_id, identity_id):
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            return (
                await conn.execute(
                    text(
                        "SELECT id FROM memberships WHERE tenant_id = :tid AND identity_id = :iid"
                    ),
                    {"tid": tenant_id, "iid": identity_id},
                )
            ).scalar_one()
    finally:
        await engine.dispose()


async def _seed_every_tenant_table(url: str, *, tenant_id, identity_id, membership_id) -> None:
    """Insert one row into every table `app.db.tenant_tables.TENANT_TABLES` names, except
    `memberships` -- `create_tenant` already wrote the admin membership this helper's own foreign
    keys (`membership_id`) reuse. As the superuser: bypasses RLS entirely, exactly like `_seed` in
    `tests/test_rls_integration.py`, so this helper works unmodified against either the pooled
    database or a dedicated tenant's own database.

    `agent_credentials.identity_id` gets its own identity and `agent`-role membership here,
    distinct from `identity_id` (the tenant's admin): migration 0041's trigger (review of #46)
    refuses an `agent_credentials` row whose `identity_id` is not an `agent`-role membership of
    the same tenant, and the admin's own membership is `admin`-role, not `agent`."""
    engine = create_async_engine(url)
    agent_identity_id = uuid.uuid4()
    try:
        async with engine.begin() as conn:
            # documents.created_by/updated_by (migration 0010) default from the
            # app.identity_id GUC -- set it (and app.tenant_id, for good measure/consistency)
            # even though this connection is the superuser and bypasses RLS entirely.
            await conn.execute(
                text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": str(tenant_id)}
            )
            await conn.execute(
                text("SELECT set_config('app.identity_id', :iid, true)"),
                {"iid": str(identity_id)},
            )
            await conn.execute(
                text(
                    "INSERT INTO documents (tenant_id, title, content) VALUES (:tid, 'Doc', 'Body')"
                ),
                {"tid": tenant_id},
            )
            await conn.execute(
                text(
                    "INSERT INTO control.identities (id, issuer, subject, kind) "
                    "VALUES (:aid, 'seed', :sub, 'agent')"
                ),
                {"aid": agent_identity_id, "sub": str(agent_identity_id)},
            )
            await conn.execute(
                text(
                    "INSERT INTO memberships (tenant_id, identity_id, role) "
                    "VALUES (:tid, :aid, 'agent')"
                ),
                {"tid": tenant_id, "aid": agent_identity_id},
            )
            await conn.execute(
                text(
                    "INSERT INTO agent_credentials "
                    "(tenant_id, identity_id, name, public_id, secret_hash, created_by) "
                    "VALUES (:tid, :aid, 'Agent', 'pub-1', 'hash-1', :iid)"
                ),
                {"tid": tenant_id, "aid": agent_identity_id, "iid": identity_id},
            )
            await conn.execute(
                text(
                    "INSERT INTO conversations (tenant_id, conversation_id, created_by) "
                    "VALUES (:tid, 'conv-1', :iid)"
                ),
                {"tid": tenant_id, "iid": identity_id},
            )
            await conn.execute(
                text(
                    "INSERT INTO messages "
                    "(tenant_id, conversation_id, sequence, payload, created_by) "
                    "VALUES (:tid, 'conv-1', 1, '{}'::jsonb, :iid)"
                ),
                {"tid": tenant_id, "iid": identity_id},
            )
            await conn.execute(
                text(
                    "INSERT INTO pending_actions "
                    "(tenant_id, conversation_id, tool_name, tool_call_id, args_hash, "
                    "asking_membership_id, expires_at) "
                    "VALUES (:tid, 'conv-1', 'a_tool', 'call-1', 'hash', :mid, "
                    "now() + interval '1 hour')"
                ),
                {"tid": tenant_id, "mid": membership_id},
            )
            await conn.execute(
                text(
                    "INSERT INTO standing_grants "
                    "(tenant_id, agent_membership_id, tool_name, granted_by) "
                    "VALUES (:tid, :mid, 'a_tool', :mid)"
                ),
                {"tid": tenant_id, "mid": membership_id},
            )
            await conn.execute(
                text(
                    "INSERT INTO approval_audit_events "
                    "(tenant_id, kind, tool_name, actor_membership_id) "
                    "VALUES (:tid, 'requested', 'a_tool', :mid)"
                ),
                {"tid": tenant_id, "mid": membership_id},
            )
    finally:
        await engine.dispose()


async def _row_counts(url: str, *, tenant_id) -> dict[str, int]:
    from app.db.tenant_tables import TENANT_TABLES

    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            counts = {}
            counts["tenants"] = (
                await conn.execute(
                    text("SELECT count(*) FROM tenants WHERE id = :tid"), {"tid": tenant_id}
                )
            ).scalar_one()
            for table in TENANT_TABLES:
                counts[table] = (
                    await conn.execute(
                        text(f"SELECT count(*) FROM {table} WHERE tenant_id = :tid"),
                        {"tid": tenant_id},
                    )
                ).scalar_one()
            counts["control.tenants"] = (
                await conn.execute(
                    text("SELECT count(*) FROM control.tenants WHERE tenant_id = :tid"),
                    {"tid": tenant_id},
                )
            ).scalar_one()
    finally:
        await engine.dispose()
    return counts


async def _tenant_erasures(url: str, *, tenant_id) -> list:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            return (
                await conn.execute(
                    text(
                        "SELECT id, tenant_id, details FROM control.tenant_erasures "
                        "WHERE tenant_id = :tid ORDER BY erased_at"
                    ),
                    {"tid": tenant_id},
                )
            ).all()
    finally:
        await engine.dispose()


async def test_erase_refuses_to_run_against_a_non_suspended_tenant(
    pooled_urls, operator_env, tmp_path
):
    """Acceptance: erase refuses to run against a tenant that is not currently suspended,
    changing nothing."""
    settings = Settings(gateway_credentials_dir=str(tmp_path))

    # A fresh engine (and disposal) per logical operator invocation, exactly as a real
    # deployment gets one -- each is its own OS process/connection, never reused. Custom GUCs
    # such as `app.tenant_id` reset to NULL only on a brand-new backend connection; reusing a
    # pooled connection across what should be separate invocations resets them to '' instead
    # (a Postgres placeholder-GUC quirk once a custom setting has been touched at all), which
    # then fails the `::uuid` cast in `control.enumerate_tenants()`'s RLS policy.
    engine = create_async_engine(pooled_urls["migrations"])
    try:
        async with engine.begin() as conn:
            created = await create_tenant(
                conn,
                tenant_name="Active Co",
                residency="eu",
                admin_email="admin@active.test",
                settings=settings,
                admin_client=_fake_admin_client(),
            )
    finally:
        await engine.dispose()

    engine2 = create_async_engine(pooled_urls["migrations"])
    try:
        async with engine2.begin() as conn:
            with pytest.raises(TenantNotSuspendedError):
                await erase_tenant(
                    conn,
                    str(created.tenant_id),
                    settings=settings,
                    trace_deleter=_fake_trace_deleter(),
                )
    finally:
        await engine2.dispose()

    counts = await _row_counts(pooled_urls["superuser"], tenant_id=created.tenant_id)
    assert counts["tenants"] == 1
    assert counts["control.tenants"] == 1
    assert counts["memberships"] == 1


async def test_erase_dry_run_reports_without_changing_anything(pooled_urls, operator_env, tmp_path):
    """Acceptance: `--dry-run` reports every step it would take against a suspended tenant and
    leaves every row, file, and credential untouched."""
    settings = Settings(gateway_credentials_dir=str(tmp_path))
    admin_client = _fake_admin_client(key="sk-dry-run")
    trace_deleter = _fake_trace_deleter()

    engine = create_async_engine(pooled_urls["migrations"])
    try:
        async with engine.begin() as conn:
            created = await create_tenant(
                conn,
                tenant_name="Dry Run Co",
                residency="eu",
                admin_email="admin@dryrun.test",
                settings=settings,
                admin_client=_fake_admin_client(key="sk-dry-run"),
            )
    finally:
        await engine.dispose()

    # A fresh engine per logical operator invocation -- see the comment in
    # test_erase_refuses_to_run_against_a_non_suspended_tenant for why.
    engine2 = create_async_engine(pooled_urls["migrations"])
    try:
        async with engine2.begin() as conn:
            await set_tenant_suspended(conn, str(created.tenant_id), suspended=True)
    finally:
        await engine2.dispose()

    secret_path = tmp_path / created.gateway_credential_alias
    assert secret_path.exists()

    engine3 = create_async_engine(pooled_urls["migrations"])
    try:
        async with engine3.begin() as conn:
            result = await erase_tenant(
                conn,
                str(created.tenant_id),
                dry_run=True,
                settings=settings,
                admin_client=admin_client,
                trace_deleter=trace_deleter,
            )
    finally:
        await engine3.dispose()

    assert isinstance(result, EraseResult)
    assert result.dry_run is True
    assert result.backup_horizon is None
    step_names = {s.step for s in result.steps}
    assert {"gateway_credential", "traces", "tenant_row"} <= step_names
    assert all("would" in s.outcome for s in result.steps)

    # Nothing touched.
    assert admin_client.delete_calls["delete_calls"] == 0
    assert trace_deleter.calls == []
    assert secret_path.exists()
    counts = await _row_counts(pooled_urls["superuser"], tenant_id=created.tenant_id)
    assert counts["tenants"] == 1
    assert counts["memberships"] == 1


async def test_erase_pooled_tenant_removes_every_registered_table_and_records_erasure(
    pooled_urls, operator_env, tmp_path, monkeypatch
):
    """Acceptance: erasing a suspended pooled tenant seeded with rows in every registered tenant
    table leaves no such rows behind (via cascade), deletes the secret file, invokes the fake
    gateway revocation and the stub trace deletion, and writes an erasure record with no foreign
    key to the tenant plus (via the real CLI dispatch) an operator-action-log entry."""
    import app.gateway_provisioning as gateway_provisioning_module
    import app.operator.erase as erase_module
    from app import config
    from app.operator.cli import _run, build_parser

    settings = Settings(gateway_credentials_dir=str(tmp_path))
    admin_client = _fake_admin_client(key="sk-pooled-erase")

    engine = create_async_engine(pooled_urls["migrations"])
    try:
        async with engine.begin() as conn:
            created = await create_tenant(
                conn,
                tenant_name="Erase Pooled Co",
                residency="eu",
                admin_email="admin@erasepooled.test",
                settings=settings,
                admin_client=admin_client,
            )
    finally:
        await engine.dispose()

    # A fresh engine per logical operator invocation -- see the comment in
    # test_erase_refuses_to_run_against_a_non_suspended_tenant for why.
    suspend_engine = create_async_engine(pooled_urls["migrations"])
    try:
        async with suspend_engine.begin() as conn:
            await set_tenant_suspended(conn, str(created.tenant_id), suspended=True)
    finally:
        await suspend_engine.dispose()

    membership_id = await _membership_id(
        pooled_urls["superuser"], tenant_id=created.tenant_id, identity_id=created.identity_id
    )
    await _seed_every_tenant_table(
        pooled_urls["superuser"],
        tenant_id=created.tenant_id,
        identity_id=created.identity_id,
        membership_id=membership_id,
    )

    before = await _row_counts(pooled_urls["superuser"], tenant_id=created.tenant_id)
    assert all(count >= 1 for count in before.values())

    secret_path = tmp_path / created.gateway_credential_alias
    assert secret_path.exists()

    # Erase through the real CLI dispatch, so this also proves the operator-action-log entry
    # (app.operator.cli._run writes it in its own `finally`, same as every other command). The
    # CLI itself takes no admin_client/trace_deleter arguments (those are internal test seams,
    # not something a real operator invocation would ever override), so the two not-yet-real
    # backends -- the gateway and Spec 8's tracing deletion -- are monkeypatched at their own
    # module-level names instead, exactly where `app.operator.erase.erase_tenant`'s defaults
    # resolve them.
    monkeypatch.setenv("GATEWAY_CREDENTIALS_DIR", str(tmp_path))
    config.get_settings.cache_clear()
    trace_deleter = _fake_trace_deleter()
    monkeypatch.setattr(erase_module, "delete_tenant_traces", trace_deleter)
    monkeypatch.setattr(
        gateway_provisioning_module, "build_admin_client", lambda settings: admin_client
    )

    args = build_parser().parse_args(["erase", str(created.tenant_id)])
    exit_code = await _run(args.command, args)
    config.get_settings.cache_clear()

    assert exit_code == 0
    assert trace_deleter.calls == [created.tenant_id]
    assert admin_client.delete_calls["delete_calls"] == 1
    assert not secret_path.exists()

    after = await _row_counts(pooled_urls["superuser"], tenant_id=created.tenant_id)
    assert all(count == 0 for count in after.values())

    erasures = await _tenant_erasures(pooled_urls["superuser"], tenant_id=created.tenant_id)
    assert len(erasures) == 1
    _, erasure_tenant_id, details = erasures[0]
    assert erasure_tenant_id == created.tenant_id
    assert details["backup_horizon"] is not None
    assert details["steps"]["gateway_credential"] == "removed"
    assert details["steps"]["traces"] == "requested"
    assert details["steps"]["tenant_row"] == "removed"

    rows = await _operator_actions(pooled_urls["superuser"], "erase")
    assert len(rows) >= 1
    tenant_id_col, action, action_details, performed_at = rows[-1]
    assert tenant_id_col == created.tenant_id
    assert action == "erase"
    assert action_details["outcome"].startswith("ok: erased")
    assert performed_at is not None


async def _operator_actions(superuser_url: str, action: str) -> list:
    """Mirrors `tests/test_operator_tool_integration.py`'s own helper: only the superuser can
    read `control.operator_actions` -- `app_owner` holds INSERT-only on it by grant."""
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
    return rows


async def test_erase_reruns_after_a_simulated_partial_failure(pooled_urls, operator_env, tmp_path):
    """Acceptance: the erasure record names a backup-horizon date, and re-running erase after a
    simulated partial failure completes the remaining steps without re-erroring on what is
    already removed."""
    settings = Settings(gateway_credentials_dir=str(tmp_path))
    flaky_admin_client = _fake_admin_client(key="sk-flaky", fail_deletes_until=1)

    engine = create_async_engine(pooled_urls["migrations"])
    try:
        async with engine.begin() as conn:
            created = await create_tenant(
                conn,
                tenant_name="Flaky Erase Co",
                residency="eu",
                admin_email="admin@flakyerase.test",
                settings=settings,
                admin_client=_fake_admin_client(key="sk-flaky"),
            )
    finally:
        await engine.dispose()

    # A fresh engine per logical operator invocation -- see the comment in
    # test_erase_refuses_to_run_against_a_non_suspended_tenant for why.
    suspend_engine = create_async_engine(pooled_urls["migrations"])
    try:
        async with suspend_engine.begin() as conn:
            await set_tenant_suspended(conn, str(created.tenant_id), suspended=True)
    finally:
        await suspend_engine.dispose()

    trace_deleter = _fake_trace_deleter()

    engine2 = create_async_engine(pooled_urls["migrations"])
    try:
        async with engine2.begin() as conn:
            first = await erase_tenant(
                conn,
                str(created.tenant_id),
                settings=settings,
                admin_client=flaky_admin_client,
                trace_deleter=trace_deleter,
            )
            await record_erasure(conn, first)
    finally:
        await engine2.dispose()

    assert first.any_step_failed
    steps_by_name = {s.step: s.outcome for s in first.steps}
    assert steps_by_name["gateway_credential"].startswith("failed:")
    assert steps_by_name["traces"] == "requested"
    assert steps_by_name["tenant_row"].startswith("skipped:")

    # The tenant is still fully present -- the failed step never let the tenant row be deleted.
    counts_after_first = await _row_counts(pooled_urls["superuser"], tenant_id=created.tenant_id)
    assert counts_after_first["tenants"] == 1

    engine3 = create_async_engine(pooled_urls["migrations"])
    try:
        async with engine3.begin() as conn:
            second = await erase_tenant(
                conn,
                str(created.tenant_id),
                settings=settings,
                admin_client=flaky_admin_client,
                trace_deleter=trace_deleter,
            )
            await record_erasure(conn, second)
    finally:
        await engine3.dispose()

    assert not second.any_step_failed
    steps_by_name_2 = {s.step: s.outcome for s in second.steps}
    assert steps_by_name_2["gateway_credential"] == "removed"
    assert steps_by_name_2["tenant_row"] == "removed"
    assert second.backup_horizon is not None

    counts_after_second = await _row_counts(pooled_urls["superuser"], tenant_id=created.tenant_id)
    assert all(count == 0 for count in counts_after_second.values())

    erasures = await _tenant_erasures(pooled_urls["superuser"], tenant_id=created.tenant_id)
    assert len(erasures) == 2  # both the partial and the completing run are recorded


async def test_erase_dedicated_tenant_drops_its_database_and_secret_files(
    pooled_urls, operator_env, dedicated_target, dedicated_secrets_dirs, tmp_path
):
    """Acceptance: erasing a suspended dedicated tenant seeded with rows in every registered
    tenant table leaves no such rows behind, via a dropped database."""
    settings = Settings(gateway_credentials_dir=str(tmp_path))
    admin_client = _fake_admin_client(key="sk-dedicated-erase")

    engine = create_async_engine(pooled_urls["migrations"])
    try:
        async with engine.begin() as conn:
            created = await create_tenant(
                conn,
                tenant_name="Erase Dedicated Co",
                residency="eu",
                admin_email="admin@erasededicated.test",
                isolation_tier="dedicated",
                dedicated_db_admin_url=dedicated_target["admin_url"],
                settings=settings,
                admin_client=admin_client,
            )
    finally:
        await engine.dispose()

    assert created.database_alias is not None
    dedicated_url = f"postgresql+asyncpg://postgres@/{created.database_alias}?host={dedicated_target['sockdir']}"

    membership_id = await _membership_id(
        dedicated_url, tenant_id=created.tenant_id, identity_id=created.identity_id
    )
    await _seed_every_tenant_table(
        dedicated_url,
        tenant_id=created.tenant_id,
        identity_id=created.identity_id,
        membership_id=membership_id,
    )

    migrations_secret = dedicated_secrets_dirs["migrations"] / created.database_alias
    app_secret = dedicated_secrets_dirs["app"] / created.database_alias
    assert migrations_secret.exists()
    assert app_secret.exists()

    suspend_engine = create_async_engine(pooled_urls["migrations"])
    try:
        async with suspend_engine.begin() as conn:
            await set_tenant_suspended(conn, str(created.tenant_id), suspended=True)
    finally:
        await suspend_engine.dispose()

    trace_deleter = _fake_trace_deleter()
    erase_engine = create_async_engine(pooled_urls["migrations"])
    try:
        async with erase_engine.begin() as conn:
            result = await erase_tenant(
                conn,
                str(created.tenant_id),
                dedicated_db_admin_url=dedicated_target["admin_url"],
                settings=settings,
                admin_client=admin_client,
                trace_deleter=trace_deleter,
            )
            await record_erasure(conn, result)
    finally:
        await erase_engine.dispose()

    assert not result.any_step_failed
    steps_by_name = {s.step: s.outcome for s in result.steps}
    assert steps_by_name["dedicated_database"] == "removed"
    assert steps_by_name["tenant_row"] == "removed"

    assert not migrations_secret.exists()
    assert not app_secret.exists()

    verify_engine = create_async_engine(dedicated_target["admin_url"])
    try:
        async with verify_engine.connect() as conn:
            exists = (
                await conn.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :n"),
                    {"n": created.database_alias},
                )
            ).first()
    finally:
        await verify_engine.dispose()
    assert exists is None

    counts = await _row_counts(pooled_urls["superuser"], tenant_id=created.tenant_id)
    assert counts["tenants"] == 0
    assert counts["control.tenants"] == 0

    erasures = await _tenant_erasures(pooled_urls["superuser"], tenant_id=created.tenant_id)
    assert len(erasures) == 1


async def test_erase_dedicated_tenant_without_admin_url_records_a_recoverable_failure(
    pooled_urls, operator_env, dedicated_target, dedicated_secrets_dirs, tmp_path
):
    """A dedicated tenant's database drop, unlike a pooled tenant's cascade, needs
    `--dedicated-db-admin-url`; omitting it is recorded as a failed (not a crashing) step, and the
    tenant stays resolvable so a re-run with the admin URL can finish the job."""
    settings = Settings(gateway_credentials_dir=str(tmp_path))
    admin_client = _fake_admin_client(key="sk-no-admin-url")

    engine = create_async_engine(pooled_urls["migrations"])
    try:
        async with engine.begin() as conn:
            created = await create_tenant(
                conn,
                tenant_name="No Admin Url Co",
                residency="eu",
                admin_email="admin@noadminurl.test",
                isolation_tier="dedicated",
                dedicated_db_admin_url=dedicated_target["admin_url"],
                settings=settings,
                admin_client=admin_client,
            )
    finally:
        await engine.dispose()

    # A fresh engine per logical operator invocation -- see the comment in
    # test_erase_refuses_to_run_against_a_non_suspended_tenant for why.
    suspend_engine = create_async_engine(pooled_urls["migrations"])
    try:
        async with suspend_engine.begin() as conn:
            await set_tenant_suspended(conn, str(created.tenant_id), suspended=True)
    finally:
        await suspend_engine.dispose()

    engine2 = create_async_engine(pooled_urls["migrations"])
    try:
        async with engine2.begin() as conn:
            result = await erase_tenant(
                conn,
                str(created.tenant_id),
                settings=settings,
                admin_client=admin_client,
                trace_deleter=_fake_trace_deleter(),
            )
    finally:
        await engine2.dispose()

    assert result.any_step_failed
    steps_by_name = {s.step: s.outcome for s in result.steps}
    assert steps_by_name["dedicated_database"].startswith("failed:")
    assert steps_by_name["tenant_row"].startswith("skipped:")

    counts = await _row_counts(pooled_urls["superuser"], tenant_id=created.tenant_id)
    assert counts["tenants"] == 1  # still resolvable for a re-run with the admin URL supplied
