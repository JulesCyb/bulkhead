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
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from urllib.parse import parse_qs, urlparse

import pgserver
import pytest

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
    common case) requests just this fixture."""
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

    monkeypatch.setenv("DATABASE_URL", cluster.app_url)
    monkeypatch.setenv("DATABASE_URL_MIGRATIONS", cluster.owner_url)
    _point_at_cluster()
    yield cluster
    _point_at_cluster()
