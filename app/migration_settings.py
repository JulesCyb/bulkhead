"""Owner-role connection string for offline tooling (Alembic migrations, the operator tool) only.

Deliberately a separate settings object from `app.config.Settings` (issue #14 / ADR-0011): the
owner/migrations DSN must never be a field on the configuration object the long-running API
process constructs, so a bug in `app.main` or `app.deps` can never read, log, or use the
credential that owns the database. Only `migrations/env.py` and `app/operator/cli.py` (the
operator tool, which replaces the retired `scripts/seed.py`) import this module — neither
`app.main` nor `app.deps` does.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class MigrationSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", secrets_dir="/run/secrets"
    )

    # Owner role (`app_owner`, no superuser, NOBYPASSRLS — docker/postgres/01-init.sh). Never
    # read from app.config.Settings.
    database_url_migrations: SecretStr


@lru_cache
def get_migration_settings() -> MigrationSettings:
    return MigrationSettings()
