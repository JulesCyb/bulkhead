"""Alembic environment (async, asyncpg). Runs with DATABASE_URL_MIGRATIONS (owner role) by
default -- overridable per invocation via `config.attributes["migration_database_url"]`, the
seam `scripts/migrate.py` (#76) uses to bring a *specific* database alias to head without
touching `Settings`/`MigrationSettings` or any other alias's database. `alembic upgrade head`
run directly from the CLI never sets that attribute, so it keeps migrating the pooled database
named by DATABASE_URL_MIGRATIONS exactly as before.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.db.models import Base
from app.migration_settings import get_migration_settings

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

_url_override = config.attributes.get("migration_database_url")
config.set_main_option(
    "sqlalchemy.url",
    _url_override or get_migration_settings().database_url_migrations.get_secret_value(),
)
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
