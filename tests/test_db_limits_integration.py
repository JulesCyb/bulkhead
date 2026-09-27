"""Real Postgres test (`pgserver`, via `uv sync --group dbtest`) for connection-level bounds
(Spec 7 / #55): a runaway query is cut off at the database inside the tenant transaction, and
the app role carries its own statement-timeout and connection-limit independent of that.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.db.guard import ROLE_CONNECTION_LIMIT, ROLE_STATEMENT_TIMEOUT_MS

pgserver = pytest.importorskip("pgserver")

from tests.support import cluster, environment, seed_tenant  # noqa: E402

_ = (cluster, environment)


@pytest.fixture
def statement_timeout_ms(environment, monkeypatch):
    """A short per-transaction statement timeout so a `pg_sleep` engineered to outlast it proves
    the cutoff happens inside the transaction, at the database, not in the application. Layered
    on top of `environment` rather than folded into it: only the two tests below need this
    override, and every other test on the shared cluster needs the real, unaltered setting."""
    from app import config

    monkeypatch.setenv("DB_STATEMENT_TIMEOUT_MS", "200")
    config.get_settings.cache_clear()
    yield environment
    config.get_settings.cache_clear()


async def test_runaway_query_is_cut_off_inside_the_tenant_transaction(statement_timeout_ms):
    from sqlalchemy.exc import DBAPIError

    from app.db.session import tenant_session

    tenant = await seed_tenant(statement_timeout_ms)
    with pytest.raises(DBAPIError):
        async with tenant_session(tenant.ctx("admin")) as session:
            # 200ms statement_timeout above; this sleeps far longer, so it must raise.
            await session.execute(text("SELECT pg_sleep(2)"))


async def test_normal_query_completes_unaffected(statement_timeout_ms):
    from app.db.session import tenant_session

    tenant = await seed_tenant(statement_timeout_ms)
    async with tenant_session(tenant.ctx("admin")) as session:
        result = await session.execute(text("SELECT 1"))
        assert result.scalar_one() == 1


async def test_app_role_has_its_own_statement_timeout_and_connection_limit(environment):
    """Role-level settings are independent of the per-transaction one set above (200ms): the
    role fact must reflect the deployment default (ROLE_STATEMENT_TIMEOUT_MS), not the
    request-scoped override."""
    engine = create_async_engine(environment.owner_url)
    async with engine.connect() as conn:
        rolconnlimit = (
            await conn.execute(text("SELECT rolconnlimit FROM pg_roles WHERE rolname = 'app'"))
        ).scalar_one()
        rolconfig = (
            await conn.execute(text("SELECT rolconfig FROM pg_roles WHERE rolname = 'app'"))
        ).scalar_one()
    await engine.dispose()

    assert rolconnlimit == ROLE_CONNECTION_LIMIT
    assert rolconfig is not None
    assert f"statement_timeout={ROLE_STATEMENT_TIMEOUT_MS}ms" in rolconfig
