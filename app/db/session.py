"""Async engine and tenant-bound sessions.

tenant_session() opens a transaction and sets app.tenant_id / app.identity_id via set_config
(is_local=true, valid for this transaction only). The RLS policies in migrations/ filter on it.
Without a set context, current_setting(..., true) returns NULL -> the policies block everything.

Defense in depth: the pool also wipes all session-local Postgres settings from a connection
when it is checked back in (see `_reset_session_state` below). is_local=true settings already
disappear at transaction end on their own, but a future bug that sets context at session scope
(set_config(..., false) or plain SET, then commits) would otherwise survive in the pool and
leak into whichever request happens to reuse that physical connection next.

It also sets a per-transaction `statement_timeout` (SET LOCAL, Spec 7 / #55): a runaway query
raises inside that transaction instead of holding the connection open past a configured deadline,
so it degrades only the tenant that issued it. The engine's pool size, overflow, timeout, and
recycle are explicit settings (app/config.py) rather than driver defaults.

control_session() (issue #22 / ADR-0003 / ADR-0011) opens a session with no app.tenant_id or
app.identity_id set at all -- the control-plane session mode. It is reserved for exactly the
narrow, cross-tenant control-plane reads the `app` role is granted (control.identity_lookup,
control.tenant_auth_settings()) and must never be used against a tenant's own tables: with no
tenant context set, current_setting(..., true) is NULL there too, so every tenant-scoped RLS
policy blocks all rows anyway -- but the point of this session mode is to make that the *only*
thing it can ever do, not to rely on RLS to save a misuse of it.

tenant_session() also resolves, on every call, which physical database serves `ctx.tenant_id`
(ADR-0002, Spec 10 / #75): it reads that tenant's isolation tier and database alias from the
control plane (`control.tenants_view`, always read against the pooled database -- the control
plane is never itself a dedicated tenant's data) and asks `app.db.engine_registry` for the
engine that alias names. A pooled tenant (the default; also any tenant control.tenants has no
row for at all, since ADR-0002 defaults every tenant to pooled) is served from the exact same
process-wide pooled engine and session factory as before this ticket; a dedicated tenant is
served from its own engine, built and cached by the registry from its alias's tenant-secret
file. Every caller keeps the exact same signature and transaction behaviour either way -- no
repository, tool, or agent run needs to know or change anything.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import ConnectionPoolEntry

from app.config import get_settings
from app.context import RequestContext

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _reset_session_state(dbapi_connection: object, connection_record: ConnectionPoolEntry) -> None:
    """Pool `checkin` hook: clear any session-local settings before the connection goes
    back into the pool, so no tenant context can ever ride along to the next checkout."""
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("RESET ALL")
    finally:
        cursor.close()


def get_engine() -> AsyncEngine:
    global _engine, _session_factory
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url.get_secret_value(),
            pool_pre_ping=True,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_recycle=settings.db_pool_recycle,
        )
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
        event.listen(_engine.sync_engine, "checkin", _reset_session_state)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    get_engine()
    assert _session_factory is not None
    return _session_factory


async def _resolve_tenant_alias(ctx: RequestContext) -> str:
    """The database alias that serves `ctx.tenant_id`, read fresh from the control plane on
    every call (ADR-0002, Spec 10 / #75).

    Always reads `control.tenants_view` against the pooled engine: the control plane itself is
    never sharded across dedicated databases. No control-plane row for the tenant, or an
    `isolation_tier` of `'pooled'`, both resolve to the pooled alias -- ADR-0002 defaults every
    tenant to pooled until an operator marks it dedicated, and the current test suite's tenants
    (seeded only in `public.tenants`, not `control.tenants`) rely on exactly that default.
    """
    from app.db.engine_registry import POOLED_ALIAS  # local import: avoids a circular import

    async with get_session_factory()() as session:
        async with session.begin():
            await session.execute(
                text("SELECT set_config('app.tenant_id', :tid, true)"),
                {"tid": str(ctx.tenant_id)},
            )
            row = (
                (
                    await session.execute(
                        text(
                            "SELECT isolation_tier, database_alias FROM control.tenants_view "
                            "WHERE tenant_id = :tid"
                        ),
                        {"tid": str(ctx.tenant_id)},
                    )
                )
                .mappings()
                .one_or_none()
            )

    if row is None or row["isolation_tier"] == "pooled":
        return POOLED_ALIAS
    return row["database_alias"]


@asynccontextmanager
async def tenant_session(ctx: RequestContext) -> AsyncIterator[AsyncSession]:
    """One transaction in the tenant's context, against whichever database the control plane
    currently assigns `ctx.tenant_id` to (ADR-0002, Spec 10 / #75; see module docstring).
    Commit at the end, rollback on error."""
    from app.db.engine_registry import POOLED_ALIAS, get_engine_for_alias  # avoids a cycle

    alias = await _resolve_tenant_alias(ctx)
    engine = await get_engine_for_alias(alias)
    factory = (
        get_session_factory()
        if alias == POOLED_ALIAS
        else async_sessionmaker(engine, expire_on_commit=False)
    )

    async with factory() as session:
        async with session.begin():
            await session.execute(
                text("SELECT set_config('app.tenant_id', :tid, true)"),
                {"tid": str(ctx.tenant_id)},
            )
            await session.execute(
                text("SELECT set_config('app.identity_id', :iid, true)"),
                {"iid": str(ctx.identity_id)},
            )
            # SET does not accept bind parameters in Postgres; the value is a validated int from
            # config, never user input, so interpolation here is safe.
            timeout_ms = int(get_settings().db_statement_timeout_ms)
            await session.execute(text(f"SET LOCAL statement_timeout = '{timeout_ms}ms'"))
            yield session


@asynccontextmanager
async def control_session() -> AsyncIterator[AsyncSession]:
    """One transaction, no tenant context: the control-plane session mode.

    Reserved for exactly the narrow control-plane reads app/repositories/control.py makes
    (the identity lookup by issuer+subject, a named tenant's auth settings). Never used against
    a tenant's own tables -- see module docstring.
    """
    async with get_session_factory()() as session:
        async with session.begin():
            timeout_ms = int(get_settings().db_statement_timeout_ms)
            await session.execute(text(f"SET LOCAL statement_timeout = '{timeout_ms}ms'"))
            yield session
