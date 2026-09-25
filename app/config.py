"""Central configuration. Values come from the environment or .env (see .env.example).

Tenant-specific things (model choice, prompts, limits) do NOT belong here — they live in
tenants.settings. This holds process-wide settings only.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "ai-app"
    environment: str = "dev"
    cors_origins: str = "http://localhost:3000"

    database_url: str = "postgresql+asyncpg://app:app@localhost:5432/app"
    database_url_migrations: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/app"

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

    # Per-membership request limit on the agent-facing routes (/agents/assistant/run,
    # /agents/assistant/stream, /api/chat): a single-process, best-effort backstop against a
    # stuck client or a scripted flood, keyed on (tenant_id, user_id). It is NOT the enforcement
    # of record for spend — that is the model gateway's own per-tenant budget; this only keeps a
    # single member's flood from denting that budget before the gateway ever notices, and only
    # within this one process (a multi-replica deployment needs a shared store for the same
    # guarantee — not implemented here).
    request_limit_max: int = 30
    request_limit_window_seconds: float = 60.0

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
