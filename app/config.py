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

from pydantic import BaseModel, Field, SecretStr, model_validator
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
    # secrets_dir (issue #14 / ADR-0011): in production, tenant secrets and connection strings can
    # be supplied as files under /run/secrets (one file per field name) instead of, or in addition
    # to, the process environment — a value here is still overridden by the matching environment
    # variable if both are present. This lets an operator rotate a secret by replacing a file and
    # redeploying, with no code change.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", secrets_dir="/run/secrets"
    )

    app_name: str = "ai-app"
    # No default (issue #14 / ADR-0011): a `.env` copied and left unedited must fail to construct
    # rather than silently becoming a permissive local configuration.
    environment: str
    cors_origins: str = "http://localhost:3000"

    # SecretStr (issue #14 / ADR-0011): every tenant-secret and connection-string field is held
    # this way so the object's own repr/str never prints a live value — read with
    # .get_secret_value() only at the one call site that needs the plain value.
    # This is the *pooled* alias's connection string only (ADR-0002, `app.db.engine_registry`,
    # Spec 10 / #75) -- the one database every tenant is served from until an operator marks it
    # `dedicated`. It is never the place for a second tenant's (or any dedicated tenant's)
    # connection details: those live one-per-alias in tenant-secret files under
    # `TENANT_DB_SECRETS_DIR` (ADR-0011), resolved lazily by `app.db.engine_registry`, never as a
    # field on this class.
    database_url: SecretStr = SecretStr("postgresql+asyncpg://app:app@localhost:5432/app")
    # The owner/migrations connection string is NOT a field here (issue #14 / ADR-0011): it is
    # removed from the application's configuration object entirely, so no code path in the
    # long-running API process can ever construct a connection with the database owner's
    # privileges. Alembic's own environment module and scripts/seed.py resolve it from
    # app.migration_settings instead — a source app.main and app.deps never import.

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
    litellm_api_key: SecretStr | None = None
    # The gateway's admin credential (Spec 7 / #53, ADR-0009): mints and revokes per-tenant
    # virtual keys through app/gateway_provisioning.py. Distinct from litellm_api_key above,
    # which the application uses to *call* the gateway as a tenant's own client would -- this
    # one is never used to build a chat/embedding client, only by the provisioning module.
    litellm_master_key: SecretStr | None = None
    # No default: a deployment with no embedding provider/model configured must refuse to
    # construct rather than silently reaching some default endpoint (ADR-0008).
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimensions: int = 1536
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    # The deployment's (or, at a call site resolving a tenant's own setting, that tenant's)
    # residency. Must be a key of RESIDENCY_ALLOW_LIST — validated below.
    residency: str = "eu"

    # No default (issue #14 / ADR-0011): same reasoning as `environment` above — a deployment
    # must choose an auth mode explicitly rather than silently running header-based tenant
    # impersonation under what looks like production.
    auth_mode: str = Field(pattern="^(dev-headers|jwt)$")

    langfuse_host: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None

    mcp_tenant_id: str | None = None
    mcp_identity_id: str | None = None

    # The process-wide token issuer used for every tenant whose control-plane
    # `identity_issuer` column is unset (issue #22 / ADR-0003): the interim "one operator-run
    # identity provider" case, open until a tenant brings its own. Read by
    # TenantAuthSettingsRepository.get() as the fallback default, never by anything under a
    # tenant's own context.
    default_identity_issuer: str | None = None

    # JWT verification (issue #24 / ADR-0003 / ADR-0012): the interim "one operator-run identity
    # provider" case (same scope as `default_identity_issuer` above) -- a single process-wide
    # verification key/algorithm, not a per-tenant or per-issuer JWKS lookup. A customer-owned
    # identity provider (still an open question per ADR-0003) needs a real JWKS-backed key
    # source; swap `app.deps.get_key_source`, never edit `app/jwt_verifier.py`, whose one job is
    # verifying a token against whatever key that dependency hands it. No default: AUTH_MODE=jwt
    # with no key configured fails every request as unauthenticated (app/deps.py) rather than
    # silently accepting unverifiable tokens.
    jwt_verification_key: SecretStr | None = None
    jwt_algorithm: str = "RS256"

    # Per-membership request limit on the agent-facing routes
    # (/v1/t/{tenant_id}/agents/assistant/run, /v1/t/{tenant_id}/agents/assistant/stream,
    # /v1/t/{tenant_id}/api/chat): a single-process, best-effort backstop against a
    # stuck client or a scripted flood, keyed on (tenant_id, identity_id). It is NOT the enforcement
    # of record for spend — that is the model gateway's own per-tenant budget; this only keeps a
    # single member's flood from denting that budget before the gateway ever notices, and only
    # within this one process (a multi-replica deployment needs a shared store for the same
    # guarantee — not implemented here).
    request_limit_max: int = 30
    request_limit_window_seconds: float = 60.0

    # Directory holding one file per gateway-credential alias (Spec 7 / #52, ADR-0009,
    # ADR-0011). Deliberately a separate field from `secrets_dir` above: `secrets_dir` is
    # pydantic-settings' own mechanism for loading *this object's own fields* once at process
    # startup (one file per field name); this directory is read fresh on every request by
    # app.gateway_credentials, keyed by an alias the control plane names per tenant, not by a
    # field name. Same default location (files delivered by the deployment under /run/secrets),
    # different lookup key and lifetime.
    gateway_credentials_dir: str = "/run/secrets"

    # Default budget and rate limit a newly provisioned tenant's gateway credential is minted
    # with (Spec 7 / #53, ADR-0009): starting defaults meant to be tuned per deployment, not
    # load-bearing constants -- a future operator tool (Spec 9) may accept per-tenant overrides
    # instead of always using these.
    gateway_default_spend_ceiling_usd: float = 50.0
    gateway_default_budget_reset_period: str = "30d"
    gateway_default_requests_per_minute: int = 60
    gateway_default_tokens_per_minute: int = 100_000

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

    # Run limits (ADR-0009, CONTEXT.md "Run limit"): the ceiling of model requests, tool calls,
    # and wall-clock time a single agent run may consume, enforced in the application — never a
    # budget, that lives at the gateway. "Order of ten model requests and twenty tool calls" is a
    # starting default meant to be tuned per deployment, not a load-bearing constant.
    run_request_limit: int = 10
    run_tool_calls_limit: int = 20
    run_total_tokens_limit: int | None = None
    run_deadline_seconds: float = 60.0

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
