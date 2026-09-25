"""Central configuration. Values come from the environment or .env (see .env.example).

Tenant-specific things (model choice, prompts, limits) do NOT belong here — they live in
tenants.settings. This holds process-wide settings only.

Residency allow-list (ADR-0008): each entry in ``RESIDENCY_ALLOW_LIST`` below names the
host patterns and endpoints one residency (jurisdiction) may reach on any content-bearing
path — model/gateway hosts, the embedding endpoint, the trace sink. Adding a residency is a
new entry in that dict, not a code change scattered across the model-routing, embeddings, and
observability modules. ``Settings.residency`` (the deployment's own residency, or a tenant's
``tenants.settings["residency"]`` at the call sites that resolve it) is validated against this
allow-list's keys, so an unknown residency identifier is rejected here, not discovered later at
request time. This module only defines the allow-list and validates identifiers against it; the
startup walk that checks every configured endpoint against its residency's allow-list, and the
resolver that returns a route for a given residency, are built in the startup-validation and
residency-resolution modules that consume this data.
"""

from functools import lru_cache

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Role-level settings for the `app` role (Spec 7 / #55): applied once in
# docker/postgres/01-init.sh, mirrored here so the embedded-Postgres integration test can assert
# them without duplicating literals. Independent of Settings.db_statement_timeout_ms below, which
# is the per-transaction timeout the application sets on every tenant_session().
ROLE_STATEMENT_TIMEOUT_MS = 60_000
ROLE_CONNECTION_LIMIT = 50


class ResidencyRoute(BaseModel):
    """The endpoints one residency may reach on a content-bearing path."""

    model_host_patterns: tuple[str, ...]
    embedding_endpoint: str
    trace_sink_host: str


# Data-driven residency allow-list. This is configuration data, not per-provider branching in
# code: a new residency is a new key here. One embedding model family is shared across every
# residency (the vector column's dimension is fixed) — residency only ever picks the region an
# endpoint is served from, never a different embedding model.
RESIDENCY_ALLOW_LIST: dict[str, ResidencyRoute] = {
    "eu": ResidencyRoute(
        model_host_patterns=("*.anthropic.com", "*.openai.com", "*.eu.litellm.internal"),
        embedding_endpoint="https://api.openai.com/v1",
        trace_sink_host="eu.cloud.langfuse.com",
    ),
    "us": ResidencyRoute(
        model_host_patterns=("*.anthropic.com", "*.openai.com", "*.us.litellm.internal"),
        embedding_endpoint="https://api.openai.com/v1",
        trace_sink_host="us.cloud.langfuse.com",
    ),
}


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
    # No default: a deployment with no embedding provider/model configured must refuse to
    # construct rather than silently reaching some default endpoint (ADR-0008).
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimensions: int = 1536
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # The deployment's (or, at a call site resolving a tenant's own setting, that tenant's)
    # residency. Must be a key of RESIDENCY_ALLOW_LIST — validated below.
    residency: str = "eu"

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

    @model_validator(mode="after")
    def _require_embedding_config(self) -> "Settings":
        if not self.embedding_provider:
            raise ValueError(
                "EMBEDDING_PROVIDER is required and has no default — set it explicitly "
                "(see .env.example) so no deployment silently reaches a default endpoint."
            )
        if not self.embedding_model:
            raise ValueError(
                "EMBEDDING_MODEL is required and has no default — set it explicitly "
                "(see .env.example) so no deployment silently reaches a default endpoint."
            )
        return self

    @model_validator(mode="after")
    def _require_known_residency(self) -> "Settings":
        if self.residency not in RESIDENCY_ALLOW_LIST:
            raise ValueError(
                f"Unknown residency {self.residency!r}. Configured residencies: "
                f"{sorted(RESIDENCY_ALLOW_LIST)} (see RESIDENCY_ALLOW_LIST in app/config.py)."
            )
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def residency_route(self) -> ResidencyRoute:
        """The allow-listed endpoints for this settings object's own residency."""
        return RESIDENCY_ALLOW_LIST[self.residency]


@lru_cache
def get_settings() -> Settings:
    return Settings()
