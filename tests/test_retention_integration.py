"""Real isolation test for the conversation retention job (ADR-0006, Spec 4 / #35) against
PostgreSQL + pgvector (`pgserver`, via `uv sync --group dbtest`). Pattern:
`tests/test_standing_grants_integration.py`.

Proves what a unit test on `ConversationsRepository.delete_expired` alone cannot (that contract
is already covered by `tests/test_rls_integration.py`): that `app/retention.py`'s job actually
visits every tenant the control plane knows about, applies each tenant's *own* retention cutoff
(its own `settings["retention_days"]`, or the documented default), deletes only what that one
tenant's own transaction is allowed to touch, and never issues a single statement against
`conversations`/`messages` on the connection it uses only to enumerate tenants.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import create_async_engine

from app.db.guard import ROLE_STATEMENT_TIMEOUT_MS  # noqa: E402
from app.tenant_settings import DEFAULT_RETENTION_DAYS  # noqa: E402

pgserver = pytest.importorskip("pgserver")

import scripts.retention as retention_script  # noqa: E402
from app.retention import run_retention_job  # noqa: E402


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
    """Mirrors `tests/test_rls_integration.py`'s own fixture: app_owner/app roles, migrated to
    head with the real Alembic chain (including 0020_conversations_and_messages)."""
    pgdata = tempfile.mkdtemp(prefix="pgdata-retention-")
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
def app_settings(database_urls, monkeypatch):
    from app import config
    from app.db import session as db_session

    monkeypatch.setenv("DATABASE_URL", database_urls["app"])
    monkeypatch.setenv("DATABASE_URL_MIGRATIONS", database_urls["migrations"])
    config.get_settings.cache_clear()
    db_session._engine = None
    db_session._session_factory = None
    yield
    config.get_settings.cache_clear()
    db_session._engine = None
    db_session._session_factory = None


async def _seed_tenant(url: str, *, name: str, retention_days: int | None = None) -> uuid.UUID:
    """A `public.tenants` row plus the matching `control.tenants` row (pooled tier, the
    default) that `control.enumerate_tenants()` -- and therefore `list_tenants`/the retention
    job -- requires to see this tenant at all (0012's INNER JOIN across both tables)."""
    engine = create_async_engine(url)
    tenant_id = uuid.uuid4()
    settings = {"retention_days": retention_days} if retention_days is not None else {}
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO tenants (id, name, settings) "
                "VALUES (:id, :name, CAST(:settings AS jsonb))"
            ),
            {"id": tenant_id, "name": name, "settings": json.dumps(settings)},
        )
        await conn.execute(
            text("INSERT INTO control.tenants (tenant_id) VALUES (:id)"), {"id": tenant_id}
        )
    await engine.dispose()
    return tenant_id


async def _seed_identity(url: str) -> uuid.UUID:
    engine = create_async_engine(url)
    identity_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO control.identities (id, issuer, subject) VALUES (:id, 'seed', :sub)"),
            {"id": identity_id, "sub": str(identity_id)},
        )
    await engine.dispose()
    return identity_id


async def _seed_conversation(
    url: str,
    *,
    tenant_id: uuid.UUID,
    identity_id: uuid.UUID,
    conversation_id: str,
    last_activity_at: datetime,
) -> None:
    """One conversation with one message, backdated directly (as the superuser, bypassing RLS by
    role -- fixture setup only, exactly like `tests/test_rls_integration.py`'s own inserts)."""
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO conversations "
                "(tenant_id, conversation_id, created_by, created_at, last_activity_at) "
                "VALUES (:tid, :cid, :iid, :ts, :ts)"
            ),
            {"tid": tenant_id, "cid": conversation_id, "iid": identity_id, "ts": last_activity_at},
        )
        await conn.execute(
            text(
                "INSERT INTO messages (tenant_id, conversation_id, sequence, payload, created_by) "
                "VALUES (:tid, :cid, 1, '{}'::jsonb, :iid)"
            ),
            {"tid": tenant_id, "cid": conversation_id, "iid": identity_id},
        )
    await engine.dispose()


async def _conversation_ids(url: str, tenant_id: uuid.UUID) -> list[str]:
    engine = create_async_engine(url)
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    text(
                        "SELECT conversation_id FROM conversations WHERE tenant_id = :tid "
                        "ORDER BY conversation_id"
                    ),
                    {"tid": tenant_id},
                )
            )
            .scalars()
            .all()
        )
    await engine.dispose()
    return list(rows)


async def _message_count(url: str, *, tenant_id: uuid.UUID, conversation_id: str) -> int:
    engine = create_async_engine(url)
    async with engine.connect() as conn:
        count = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM messages "
                    "WHERE tenant_id = :tid AND conversation_id = :cid"
                ),
                {"tid": tenant_id, "cid": conversation_id},
            )
        ).scalar_one()
    await engine.dispose()
    return count


class _FakeSecret:
    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value


class _FakeMigrationSettings:
    def __init__(self, url: str) -> None:
        self.database_url_migrations = _FakeSecret(url)


async def test_job_deletes_only_each_tenants_own_expired_conversations(app_settings, database_urls):
    """AC1: sweeping every tenant in one job run deletes only the currently-processed tenant's
    conversations whose last activity is past its own cutoff -- that tenant's fresher
    conversations, and another tenant's conversations, are left untouched."""
    now = datetime.now(UTC)
    tenant_a = await _seed_tenant(database_urls["superuser"], name="Acme")
    tenant_b = await _seed_tenant(database_urls["superuser"], name="Globex")
    identity_a = await _seed_identity(database_urls["superuser"])
    identity_b = await _seed_identity(database_urls["superuser"])

    # Well past the documented default (90 days) -- expired for tenant A.
    await _seed_conversation(
        database_urls["superuser"],
        tenant_id=tenant_a,
        identity_id=identity_a,
        conversation_id="conv-old",
        last_activity_at=now - timedelta(days=DEFAULT_RETENTION_DAYS + 30),
    )
    # Recent -- not expired for tenant A.
    await _seed_conversation(
        database_urls["superuser"],
        tenant_id=tenant_a,
        identity_id=identity_a,
        conversation_id="conv-fresh",
        last_activity_at=now,
    )
    # Tenant B's own conversation is recent too -- must survive the sweep untouched even though
    # the job visits tenant B in the very same run.
    await _seed_conversation(
        database_urls["superuser"],
        tenant_id=tenant_b,
        identity_id=identity_b,
        conversation_id="conv-b",
        last_activity_at=now,
    )

    engine = create_async_engine(database_urls["migrations"])
    try:
        async with engine.begin() as conn:
            outcomes = await run_retention_job(conn)
    finally:
        await engine.dispose()

    by_tenant = {outcome.tenant_id: outcome.deleted for outcome in outcomes}
    assert by_tenant[tenant_a] == 1
    assert by_tenant[tenant_b] == 0

    assert await _conversation_ids(database_urls["superuser"], tenant_a) == ["conv-fresh"]
    assert await _conversation_ids(database_urls["superuser"], tenant_b) == ["conv-b"]


async def test_a_tenants_own_retention_setting_is_honored_over_the_default(
    app_settings, database_urls
):
    """AC2: a tenant that has set its own (shorter) retention period on its tenant settings is
    expired using that period as the cutoff, not the documented default -- proving the setting is
    actually honored, not just readable. 20 days old is *inside* the 90-day default window (would
    survive under the default) but past this tenant's own 10-day setting."""
    now = datetime.now(UTC)
    tenant = await _seed_tenant(
        database_urls["superuser"], name="ShortRetention", retention_days=10
    )
    identity = await _seed_identity(database_urls["superuser"])
    await _seed_conversation(
        database_urls["superuser"],
        tenant_id=tenant,
        identity_id=identity,
        conversation_id="conv-20-days-old",
        last_activity_at=now - timedelta(days=20),
    )

    engine = create_async_engine(database_urls["migrations"])
    try:
        async with engine.begin() as conn:
            outcomes = await run_retention_job(conn)
    finally:
        await engine.dispose()

    assert {o.tenant_id: o.deleted for o in outcomes}[tenant] == 1
    assert await _conversation_ids(database_urls["superuser"], tenant) == []


async def test_job_leaves_no_orphaned_messages_behind(app_settings, database_urls):
    """AC3: deleting an expired conversation removes its messages with it -- no orphaned message
    rows survive the sweep."""
    now = datetime.now(UTC)
    tenant = await _seed_tenant(database_urls["superuser"], name="Initech")
    identity = await _seed_identity(database_urls["superuser"])
    await _seed_conversation(
        database_urls["superuser"],
        tenant_id=tenant,
        identity_id=identity,
        conversation_id="conv-expired",
        last_activity_at=now - timedelta(days=DEFAULT_RETENTION_DAYS + 1),
    )

    engine = create_async_engine(database_urls["migrations"])
    try:
        async with engine.begin() as conn:
            await run_retention_job(conn)
    finally:
        await engine.dispose()

    count = await _message_count(
        database_urls["superuser"], tenant_id=tenant, conversation_id="conv-expired"
    )
    assert count == 0


async def test_job_never_touches_conversations_or_messages_on_the_enumeration_connection(
    app_settings, database_urls
):
    """AC4: the connection passed to `run_retention_job` (an `app_owner` connection) is used only
    to enumerate tenants -- every statement it issues names `control.enumerate_tenants()` and
    nothing else. The per-tenant deletion always happens on a separate connection, opened by
    `tenant_session(ctx)` (the ordinary, RLS-scoped `app`-role path)."""
    now = datetime.now(UTC)
    tenant = await _seed_tenant(database_urls["superuser"], name="Umbrella")
    identity = await _seed_identity(database_urls["superuser"])
    await _seed_conversation(
        database_urls["superuser"],
        tenant_id=tenant,
        identity_id=identity,
        conversation_id="conv-expired",
        last_activity_at=now - timedelta(days=DEFAULT_RETENTION_DAYS + 1),
    )

    engine = create_async_engine(database_urls["migrations"])
    statements: list[str] = []

    def _capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", _capture)
    try:
        async with engine.begin() as conn:
            await run_retention_job(conn)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", _capture)
        await engine.dispose()

    assert statements, "expected at least the enumerate_tenants() call to be captured"
    for statement in statements:
        lowered = statement.lower()
        assert "conversations" not in lowered
        assert "messages" not in lowered
    assert any("enumerate_tenants" in statement.lower() for statement in statements)

    # And the deletion really did happen -- just not on this connection.
    assert await _conversation_ids(database_urls["superuser"], tenant) == []


async def test_retention_script_entry_point_runs_end_to_end(
    app_settings, database_urls, monkeypatch
):
    """The documented, runnable script (`uv run python scripts/retention.py`) actually deletes
    expired conversations end to end, through its real async entry point (`_main`, which `main()`
    only wraps in `asyncio.run` -- called directly here since this test already runs inside an
    event loop) -- only `get_migration_settings` is faked, to point it at the ephemeral test
    database instead of reading `.env`/the process environment for a real owner DSN."""
    now = datetime.now(UTC)
    tenant = await _seed_tenant(database_urls["superuser"], name="Soylent")
    identity = await _seed_identity(database_urls["superuser"])
    await _seed_conversation(
        database_urls["superuser"],
        tenant_id=tenant,
        identity_id=identity,
        conversation_id="conv-expired",
        last_activity_at=now - timedelta(days=DEFAULT_RETENTION_DAYS + 1),
    )
    await _seed_conversation(
        database_urls["superuser"],
        tenant_id=tenant,
        identity_id=identity,
        conversation_id="conv-fresh",
        last_activity_at=now,
    )

    monkeypatch.setattr(
        retention_script,
        "get_migration_settings",
        lambda: _FakeMigrationSettings(database_urls["migrations"]),
    )

    await retention_script._main()

    assert await _conversation_ids(database_urls["superuser"], tenant) == ["conv-fresh"]


async def test_a_suspended_tenants_conversations_are_untouched_by_the_job(
    app_settings, database_urls
):
    """#106, ADR-0010: a suspended tenant is skipped outright (one log line, no `tenant_session()`
    opened for it) rather than raising `TenantSuspendedError` mid-sweep -- its own expired
    conversation survives the run untouched, and it is not counted among the returned outcomes,
    while an unsuspended tenant in the very same run is swept normally."""
    now = datetime.now(UTC)
    suspended = await _seed_tenant(database_urls["superuser"], name="Suspended")
    active = await _seed_tenant(database_urls["superuser"], name="Active")
    identity_suspended = await _seed_identity(database_urls["superuser"])
    identity_active = await _seed_identity(database_urls["superuser"])
    await _seed_conversation(
        database_urls["superuser"],
        tenant_id=suspended,
        identity_id=identity_suspended,
        conversation_id="conv-suspended-expired",
        last_activity_at=now - timedelta(days=DEFAULT_RETENTION_DAYS + 30),
    )
    await _seed_conversation(
        database_urls["superuser"],
        tenant_id=active,
        identity_id=identity_active,
        conversation_id="conv-active-expired",
        last_activity_at=now - timedelta(days=DEFAULT_RETENTION_DAYS + 30),
    )

    engine = create_async_engine(database_urls["superuser"])
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE control.tenants SET suspended_at = now() WHERE tenant_id = :tid"),
            {"tid": suspended},
        )
    await engine.dispose()

    engine = create_async_engine(database_urls["migrations"])
    try:
        async with engine.begin() as conn:
            outcomes = await run_retention_job(conn)
    finally:
        await engine.dispose()

    assert suspended not in {o.tenant_id for o in outcomes}
    assert {o.tenant_id: o.deleted for o in outcomes}[active] == 1

    assert await _conversation_ids(database_urls["superuser"], suspended) == [
        "conv-suspended-expired"
    ]
    assert await _conversation_ids(database_urls["superuser"], active) == []
