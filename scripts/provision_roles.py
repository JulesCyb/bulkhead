"""Standalone role/grant provisioning for managed Postgres (RDS, Neon, Supabase, Cloud SQL) that
gives no first-boot container hook to run `docker/postgres/01-init.sh` (Spec 9 / #67).

Creates the same `app_owner` (owns every object, no superuser, no RLS bypass, runs migrations
and seeding) and `app` (the long-running application role — no superuser, no RLS bypass, a
role-level connection limit and statement timeout) roles, with the same attributes and grants,
that the container's init script creates locally. See `docker/postgres/01-init.sh` for what each
role is for; this script must keep matching it exactly.

Idempotent: running it again against an already-provisioned database creates nothing that
already exists and only re-applies grants that are no-ops when already held, so a second run
changes nothing further.

    uv run python scripts/provision_roles.py postgresql://admin:pw@host:5432/app

The positional argument is a connection URL for a role that can create roles and extensions on
the target database — the managed provider's own admin/master role, used only for this one run,
never by the application afterwards. Role passwords and connection/timeout settings come from
the same environment variables `docker/postgres/01-init.sh` reads, so one `.env` serves both:

    APP_OWNER_DB_PASSWORD  APP_DB_PASSWORD  APP_STATEMENT_TIMEOUT_MS  APP_CONNECTION_LIMIT
"""

from __future__ import annotations

import asyncio
import os
import sys

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import ROLE_CONNECTION_LIMIT, ROLE_STATEMENT_TIMEOUT_MS


def _as_asyncpg_url(url: str) -> str:
    """Accepts a plain `postgresql://` URL (what an operator pastes from their provider's
    console) or one that already names the asyncpg driver."""
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url[len("postgresql://") :]
    raise SystemExit(
        f"Unsupported URL scheme: {url!r} (expected postgresql:// or postgresql+asyncpg://)"
    )


def _sql_literal(value: str) -> str:
    """A single-quoted SQL string literal. Escapes embedded quotes; good enough for the
    operator-supplied passwords this script handles (same trust model as the shell heredoc in
    docker/postgres/01-init.sh, which interpolates these same env vars unescaped)."""
    return "'" + value.replace("'", "''") + "'"


async def provision(admin_url: str) -> None:
    url = make_url(_as_asyncpg_url(admin_url))
    dbname = url.database
    if not dbname:
        raise SystemExit(f"Admin URL is missing a database name: {admin_url!r}")

    app_owner_password = _sql_literal(os.environ.get("APP_OWNER_DB_PASSWORD", "app_owner"))
    app_password = _sql_literal(os.environ.get("APP_DB_PASSWORD", "app"))
    statement_timeout_ms = int(
        os.environ.get("APP_STATEMENT_TIMEOUT_MS", ROLE_STATEMENT_TIMEOUT_MS)
    )
    connection_limit = int(os.environ.get("APP_CONNECTION_LIMIT", ROLE_CONNECTION_LIMIT))

    engine = create_async_engine(str(url))
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

            # DO blocks: CREATE ROLE has no IF NOT EXISTS, so a fresh database creates the role
            # and a re-run alters the existing one to the same attributes/password instead —
            # neither branch changes any grant made below.
            await conn.execute(
                text(
                    f"""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'app_owner') THEN
                            CREATE ROLE app_owner LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB
                                NOCREATEROLE PASSWORD {app_owner_password};
                        ELSE
                            ALTER ROLE app_owner WITH LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB
                                NOCREATEROLE PASSWORD {app_owner_password};
                        END IF;
                    END
                    $$;
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'app') THEN
                            CREATE ROLE app LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE
                                PASSWORD {app_password} CONNECTION LIMIT {connection_limit};
                        ELSE
                            ALTER ROLE app WITH LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB
                                NOCREATEROLE PASSWORD {app_password} CONNECTION LIMIT
                                {connection_limit};
                        END IF;
                    END
                    $$;
                    """
                )
            )
            await conn.execute(
                text(f"ALTER ROLE app SET statement_timeout = '{statement_timeout_ms}ms'")
            )

            quoted_db = '"' + dbname.replace('"', '""') + '"'
            await conn.execute(text(f"GRANT CONNECT ON DATABASE {quoted_db} TO app_owner"))
            await conn.execute(text("ALTER SCHEMA public OWNER TO app_owner"))
            await conn.execute(text(f"GRANT CONNECT ON DATABASE {quoted_db} TO app"))
            await conn.execute(text("GRANT USAGE ON SCHEMA public TO app"))
    finally:
        await engine.dispose()

    print("app_owner and app roles provisioned (or already matched the desired state).")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: uv run python scripts/provision_roles.py <admin-database-url>")
    asyncio.run(provision(sys.argv[1]))


if __name__ == "__main__":
    main()
