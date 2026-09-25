"""Central configuration. Values come from the environment or .env (see .env.example).

Tenant-specific things (model choice, prompts, limits) do NOT belong here — they live in
tenants.settings. This holds process-wide settings only.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Role-level settings for the `app` role (Spec 7 / #55): applied once in
# docker/postgres/01-init.sh, mirrored here so the embedded-Postgres integration test can assert
# them without duplicating literals. Independent of Settings.db_statement_timeout_ms below, which
# is the per-transaction timeout the application sets on every tenant_session().
ROLE_STATEMENT_TIMEOUT_MS = 60_000
ROLE_CONNECTION_LIMIT = 50


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "ai-app"
    environment: str = "dev"
    cors_origins: str = "http://localhost:3000"

    database_url: str = "postgresql+asyncpg://app:app@localhost:5432/app"
    database_url_migrations: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/app"

    # Explicit pool sizing (Spec 7 / #55) — named configuration instead of SQLAlchemy/driver
    # defaults, so the deployment's real concurrency ceiling (pool_size + max_overflow, per
    # worker process) is visible in one place instead of guessed.
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout: int = 30  # seconds a checkout waits for a free connection
    db_pool_recycle: int = 1800  # seconds before a pooled connection is recycled
    # Per-transaction statement timeout (ms), set with SET LOCAL in tenant_session() so a
    # runaway query is cut off inside that tenant's transaction and the connection is freed.
    db_statement_timeout_ms: int = 30_000

    llm_model: str = "anthropic:claude-sonnet-4-5"
    litellm_base_url: str | None = None
    litellm_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    auth_mode: str = Field(default="dev-headers", pattern="^(dev-headers|jwt)$")

    langfuse_host: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None

    mcp_tenant_id: str | None = None
    mcp_user_id: str | None = None

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
