"""Async engine and tenant-bound sessions.

tenant_session() opens a transaction and sets app.tenant_id / app.user_id via set_config
(is_local=true, valid for this transaction only). The RLS policies in migrations/ filter on it.
Without a set context, current_setting(..., true) returns NULL -> the policies block everything.

It also sets a per-transaction `statement_timeout` (SET LOCAL, Spec 7 / #55): a runaway query
raises inside that transaction instead of holding the connection open past a configured deadline,
so it degrades only the tenant that issued it. The engine's pool size, overflow, timeout, and
recycle are explicit settings (app/config.py) rather than driver defaults.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings
from app.context import RequestContext

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine, _session_factory
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_recycle=settings.db_pool_recycle,
        )
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    get_engine()
    assert _session_factory is not None
    return _session_factory


@asynccontextmanager
async def tenant_session(ctx: RequestContext) -> AsyncIterator[AsyncSession]:
    """One transaction in the tenant's context. Commit at the end, rollback on error."""
    async with get_session_factory()() as session:
        async with session.begin():
            await session.execute(
                text("SELECT set_config('app.tenant_id', :tid, true)"),
                {"tid": str(ctx.tenant_id)},
            )
            await session.execute(
                text("SELECT set_config('app.user_id', :uid, true)"),
                {"uid": str(ctx.user_id)},
            )
            # SET does not accept bind parameters in Postgres; the value is a validated int from
            # config, never user input, so interpolation here is safe.
            timeout_ms = int(get_settings().db_statement_timeout_ms)
            await session.execute(text(f"SET LOCAL statement_timeout = '{timeout_ms}ms'"))
            yield session
