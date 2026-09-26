"""One embedded Postgres cluster, shared across the integration suite (issue #96 / spec #90,
"A6").

Boots `pgserver` once per test session, bootstraps the same two roles
`docker/postgres/01-init.sh` creates (`app_owner`, `app` -- see `ROLE_BOOTSTRAP_SQL` below), and
migrates the pooled database to head with the real Alembic chain -- the same runner production
uses, invoked here as a subprocess exactly like every integration file already did.

`ROLE_BOOTSTRAP_SQL` is kept as the single string every one of the (formerly seventeen)
integration files used to hand-copy, so it now exists exactly once and a future test can compare
it against `docker/postgres/01-init.sh`'s own statements without hand-copying it a second time.
It is a reduced mirror, not the script itself: no passwords (pgserver's default trust auth needs
none), pgserver's single default database rather than a named `${POSTGRES_DB}`, and no `gateway`
role or database -- the script's own `CREATE DATABASE`/`GATEWAY_DB_PASSWORD`/`REVOKE CONNECT FROM
PUBLIC` statements are exercised for real, by running the script itself, in
`tests/test_gateway_bootstrap_integration.py`.
"""

from __future__ import annotations

import dataclasses
import logging.config
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import parse_qs, urlparse

import pgserver
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import ROLE_STATEMENT_TIMEOUT_MS

ROLE_BOOTSTRAP_SQL = (
    "CREATE EXTENSION IF NOT EXISTS vector; "
    "CREATE ROLE app_owner LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE; "
    "ALTER SCHEMA public OWNER TO app_owner; "
    "GRANT CREATE ON DATABASE postgres TO app_owner; "
    "CREATE ROLE app LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE; "
    "GRANT USAGE ON SCHEMA public TO app; "
    f"ALTER ROLE app SET statement_timeout = '{ROLE_STATEMENT_TIMEOUT_MS}ms';"
)


def _run_sql(server: object, command: str) -> None:
    """`server.psql` without a shell: pgserver's own version breaks on paths with spaces."""
    from pgserver.postgres_server import POSTGRES_BIN_PATH

    subprocess.run(
        [str(POSTGRES_BIN_PATH / "psql"), server.get_uri()],
        input=command.encode(),
        check=True,
        capture_output=True,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Cluster:
    """Connection URLs for the one embedded cluster this session boots.

    `owner_url` is the `app_owner` role -- the same role production migrations and the operator
    tool connect as (`DATABASE_URL_MIGRATIONS`); seeding (`tests.support.seeding`) writes as this
    role too. `app_url` is the `app` role the running application connects as. `superuser_url`
    is the cluster's own bootstrap superuser: test-only, since `app_owner` does not bypass RLS --
    a test that needs to plant or read a row with no tenant context in scope needs the real
    superuser, exactly like production never would.
    """

    superuser_url: str
    owner_url: str
    app_url: str


@pytest.fixture(scope="session")
def cluster() -> Iterator[Cluster]:
    """One embedded Postgres cluster for the whole test session: role bootstrap once, migrations
    to head once, for the pooled alias. A dedicated alias's own database, when a later ticket
    needs one, is created on demand by the seeding function -- this fixture only ever boots the
    one pooled cluster."""
    pgdata = tempfile.mkdtemp(prefix="pgdata-")
    server = pgserver.get_server(pgdata, cleanup_mode="delete")
    sockdir = parse_qs(urlparse(server.get_uri()).query)["host"][0]
    _run_sql(server, ROLE_BOOTSTRAP_SQL)

    urls = Cluster(
        superuser_url=f"postgresql+asyncpg://postgres@/postgres?host={sockdir}",
        owner_url=f"postgresql+asyncpg://app_owner@/postgres?host={sockdir}",
        app_url=f"postgresql+asyncpg://app@/postgres?host={sockdir}",
    )
    env = {**os.environ, "DATABASE_URL_MIGRATIONS": urls.owner_url, "DATABASE_URL": urls.app_url}
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"], check=True, env=env, timeout=120
    )
    yield urls
    server.cleanup()


@pytest.fixture
def environment(cluster: Cluster, monkeypatch: pytest.MonkeyPatch) -> Iterator[Cluster]:
    """Points `Settings`, the session layer, the engine registry, and the per-tenant client
    caches at `cluster`'s pooled database for the duration of one test, and undoes all of it
    afterwards -- the cache-clear/engine-reset dance every integration file used to repeat.
    Yields `cluster` itself: a test that only needs the environment pointed at the cluster (the
    common case) requests just this fixture.

    Also points `TENANT_DB_SECRETS_DIR` (read by `app/db/engine_registry.py`, app-role DSNs) and
    `TENANT_DB_MIGRATIONS_SECRETS_DIR` (read by `scripts/migrate.py`, owner-role DSNs) at fresh
    temporary directories, removed on teardown -- so `seed_tenant(..., isolation_tier=
    "dedicated")` (`tests.support.seeding`) has somewhere real to write a dedicated alias's
    secret files, and so no test -- dedicated or not -- ever reads or writes either directory's
    production default (`/run/secrets/tenant-db(-migrations)`)."""
    import shutil

    from app import config
    from app.db import engine_registry
    from app.db import session as db_session
    from app.embeddings import reset_tenant_embedding_client_cache
    from app.llm import reset_tenant_chat_model_cache

    def _point_at_cluster() -> None:
        config.get_settings.cache_clear()
        db_session._engine = None
        db_session._session_factory = None
        engine_registry.reset_registry_for_tests()
        reset_tenant_chat_model_cache()
        reset_tenant_embedding_client_cache()

    secrets_dir = tempfile.mkdtemp(prefix="tenant-db-secrets-")
    migrations_secrets_dir = tempfile.mkdtemp(prefix="tenant-db-migrations-secrets-")

    monkeypatch.setenv("DATABASE_URL", cluster.app_url)
    monkeypatch.setenv("DATABASE_URL_MIGRATIONS", cluster.owner_url)
    monkeypatch.setenv("TENANT_DB_SECRETS_DIR", secrets_dir)
    monkeypatch.setenv("TENANT_DB_MIGRATIONS_SECRETS_DIR", migrations_secrets_dir)
    _point_at_cluster()
    try:
        yield cluster
    finally:
        _point_at_cluster()
        shutil.rmtree(secrets_dir, ignore_errors=True)
        shutil.rmtree(migrations_secrets_dir, ignore_errors=True)


def _with_database(url: str, database: str) -> str:
    """The same DSN as `url`, pointed at a different database name on the same server --swaps
    only the path segment before the query string. Never
    `sqlalchemy.engine.URL.render_as_string()`/`str(url)`: both percent-encode the unix-socket
    path carried in `?host=...`, and a literal `%` then breaks `ConfigParser`'s own
    interpolation the moment `scripts.migrate._upgrade_head` hands the DSN to Alembic (see
    `app.operator.dedicated_db._render_dsn`'s docstring, which hits the same trap)."""
    base, _, query = url.partition("?")
    prefix, _, _old_db = base.rpartition("/")
    new_base = f"{prefix}/{database}"
    return f"{new_base}?{query}" if query else new_base


async def create_database(cluster: Cluster, name: str) -> Cluster:
    """Creates a fresh, empty database named `name` on the same server `cluster` already boots,
    with the same per-database bootstrap `docker/postgres/01-init.sh` performs on its own single
    database (the `vector` extension, schema ownership, grants) -- everything `CREATE DATABASE`
    itself does not inherit from `ROLE_BOOTSTRAP_SQL`'s cluster-wide role bootstrap (roles are
    cluster-wide in Postgres; schema ownership and grants are per-database).

    Deliberately not migrated: the caller decides whether and how.
    `tests.support.seeding.seed_tenant`'s dedicated-tier path migrates it immediately with the
    real runner; `tests/test_migrate_alias_integration.py` leaves it exactly as this function
    returns it, on purpose, to prove the runner itself brings a fresh database to head.

    Returns a new `Cluster` -- same shape, same three roles, pointed at `name` instead of
    `cluster`'s own default database.
    """
    quoted = '"' + name.replace('"', '""') + '"'

    admin_engine = create_async_engine(cluster.superuser_url, isolation_level="AUTOCOMMIT")
    try:
        async with admin_engine.connect() as conn:
            await conn.execute(text(f"CREATE DATABASE {quoted}"))
    finally:
        await admin_engine.dispose()

    fresh = Cluster(
        superuser_url=_with_database(cluster.superuser_url, name),
        owner_url=_with_database(cluster.owner_url, name),
        app_url=_with_database(cluster.app_url, name),
    )

    bootstrap_engine = create_async_engine(fresh.superuser_url)
    try:
        async with bootstrap_engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(text("ALTER SCHEMA public OWNER TO app_owner"))
            await conn.execute(text(f"GRANT CREATE ON DATABASE {quoted} TO app_owner"))
            await conn.execute(text("GRANT USAGE ON SCHEMA public TO app"))
    finally:
        await bootstrap_engine.dispose()

    return fresh


@contextmanager
def migration_run_without_disrupting_logging() -> Iterator[None]:
    """Wrap any in-process call into the real migration runner (`scripts.migrate.migrate_alias`/
    `migrate_all`, or anything that itself calls `alembic.command.upgrade`) in this, every time.

    `migrations/env.py` calls `logging.config.fileConfig(config.config_file_name)` on every real
    migration run (`command.upgrade()` re-executes `env.py` fresh each call, not a one-time
    import, so this fires on every single call, not just the first). By default that disables
    every logger that already exists at that moment, process-wide -- harmless across a
    subprocess boundary (every private per-file bootstrap this package replaced shelled out to
    `alembic upgrade head` as a *subprocess*, exactly to keep this contained, and so does this
    package's own session-scoped `cluster` fixture above), but running the real migration runner
    in-process instead (this package's dedicated-tier seeding, deliberately, to exercise the
    exact code path `app.operator.dedicated_db.ensure_dedicated_database` uses) would otherwise
    silently disable module-level `log = logging.getLogger(__name__)` objects created once, at
    import time, and reused for the rest of the pytest session -- breaking every later test in
    the *same session* that asserts against `caplog` for one of them, with no exception raised
    anywhere to explain why. `env.py`'s own `from logging.config import fileConfig` re-binds
    fresh from this attribute on every run, so patching it here, only for the duration of one
    migration call, is enough -- never touches `migrations/env.py` itself.
    """
    original = logging.config.fileConfig
    logging.config.fileConfig = lambda *args, **kwargs: None
    try:
        yield
    finally:
        logging.config.fileConfig = original
