"""Real isolation test against PostgreSQL + pgvector (`pgserver`, via `uv sync --group dbtest`).

Verifies what the unit tests cannot: that the RLS policies from the migration apply when the
app works as the `app` role (no superuser, NOBYPASSRLS).
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

pgserver = pytest.importorskip("pgserver")

DIM = 1536


def _vec(seed: float) -> str:
    values = [0.0] * DIM
    values[0] = 1.0
    values[1] = seed
    return "[" + ",".join(f"{v:.3f}" for v in values) + "]"


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
    pgdata = tempfile.mkdtemp(prefix="pgdata-")
    server = pgserver.get_server(pgdata)
    sockdir = parse_qs(urlparse(server.get_uri()).query)["host"][0]
    _psql(
        server,
        "CREATE ROLE app LOGIN NOSUPERUSER NOBYPASSRLS; "
        "GRANT USAGE ON SCHEMA public TO app; "
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE "
        "ON TABLES TO app;",
    )
    urls = {
        "migrations": f"postgresql+asyncpg://postgres@/postgres?host={sockdir}",
        "app": f"postgresql+asyncpg://app@/postgres?host={sockdir}",
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


async def _seed(url: str) -> tuple[uuid.UUID, uuid.UUID]:
    """Two tenants with one document each — as the owner; without context RLS blocks all."""
    engine = create_async_engine(url)
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    async with engine.begin() as conn:
        for tenant_id, name, seed in ((tenant_a, "A", 0.1), (tenant_b, "B", 0.9)):
            await conn.execute(
                text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
                {"id": tenant_id, "name": name},
            )
            await conn.execute(
                text(
                    "INSERT INTO documents (tenant_id, title, content, embedding) "
                    "VALUES (:tid, :title, :content, CAST(:emb AS vector))"
                ),
                {
                    "tid": tenant_id,
                    "title": f"Document {name}",
                    "content": f"Content of tenant {name}",
                    "emb": _vec(seed),
                },
            )
    await engine.dispose()
    return tenant_a, tenant_b


async def test_search_sees_only_own_tenant(app_settings, database_urls):
    from app.context import RequestContext
    from app.db.session import tenant_session
    from app.repositories.documents import DocumentRepository

    tenant_a, tenant_b = await _seed(database_urls["migrations"])
    query = [0.0] * DIM
    query[0] = 1.0

    ctx_a = RequestContext(tenant_id=tenant_a, user_id=uuid.uuid4())
    async with tenant_session(ctx_a) as session:
        hits = await DocumentRepository().search(session, query, limit=10)
    assert [h.title for h in hits] == ["Document A"]

    ctx_b = RequestContext(tenant_id=tenant_b, user_id=uuid.uuid4())
    async with tenant_session(ctx_b) as session:
        hits = await DocumentRepository().search(session, query, limit=10)
    assert [h.title for h in hits] == ["Document B"]


async def test_insert_for_other_tenant_is_rejected(app_settings, database_urls):
    from sqlalchemy.exc import DBAPIError

    from app.context import RequestContext
    from app.db.session import tenant_session

    tenant_a, tenant_b = await _seed(database_urls["migrations"])
    ctx_a = RequestContext(tenant_id=tenant_a, user_id=uuid.uuid4())
    with pytest.raises(DBAPIError):
        async with tenant_session(ctx_a) as session:
            await session.execute(
                text(
                    "INSERT INTO documents (tenant_id, title, content) "
                    "VALUES (:tid, 'foreign', 'must not work')"
                ),
                {"tid": tenant_b},
            )


async def test_app_role_cannot_delete_tenants(app_settings, database_urls):
    """The migration revokes INSERT/DELETE on tenants from the app role — a DELETE would
    cascade an entire tenant away in one statement."""
    from sqlalchemy.exc import DBAPIError, ProgrammingError

    from app.context import RequestContext
    from app.db.session import tenant_session

    tenant_a, _ = await _seed(database_urls["migrations"])
    ctx_a = RequestContext(tenant_id=tenant_a, user_id=uuid.uuid4())
    with pytest.raises((DBAPIError, ProgrammingError)):
        async with tenant_session(ctx_a) as session:
            await session.execute(text("DELETE FROM tenants WHERE id = :tid"), {"tid": tenant_a})


async def test_no_context_means_no_rows(app_settings, database_urls):
    """Without set_config, current_setting is NULL -> the policy blocks all (app role)."""
    await _seed(database_urls["migrations"])
    engine = create_async_engine(database_urls["app"])
    async with engine.connect() as conn:
        count = (await conn.execute(text("SELECT count(*) FROM documents"))).scalar_one()
    await engine.dispose()
    assert count == 0


async def test_pool_checkin_clears_leftover_tenant_context(app_settings):
    """A connection returned to the pool carries none of a request's session-local
    settings. Simulates the bug this guards against: a context committed at session
    scope (set_config(..., false)), not transaction scope (is_local=true) — a plain
    rollback-on-checkin would not undo an already-committed session setting, so the
    pool must explicitly wipe it on checkin instead."""
    from app.db.session import get_engine

    engine = get_engine()
    tenant_id = str(uuid.uuid4())

    conn = await engine.connect()
    await conn.execute(text("SELECT set_config('app.tenant_id', :tid, false)"), {"tid": tenant_id})
    set_value = (
        await conn.execute(text("SELECT current_setting('app.tenant_id', true)"))
    ).scalar_one()
    assert set_value == tenant_id
    await conn.commit()  # the setting is now session-level and committed, not rolled back
    await conn.close()  # checkin

    conn2 = await engine.connect()
    try:
        leftover = (
            await conn2.execute(text("SELECT current_setting('app.tenant_id', true)"))
        ).scalar_one()
    finally:
        await conn2.close()
    assert leftover in (None, "")
