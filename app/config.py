"""Central configuration. Values come from the environment or .env (see .env.example).

Tenant-specific things (model choice, prompts, limits) do NOT belong here — they live in
tenants.settings. This module holds process-wide settings only: fields, the validators that
check them, and the cached `get_settings()` accessor (spec A4 / #94, #112) -- role-level DB
constants live next to `app.db.guard`, JWT algorithm policy in `app.jwt_verifier`, and agent-token
key derivation in `app.agent_credential_exchange`.

Residency allow-list (ADR-0008, spec A4 / #94, #110): `Settings.residency_allow_list` holds one
`app.residency.ResidencyAllowList` instance, built from `config/residency.toml`
(`RESIDENCY_CONFIG_PATH`) at `Settings` construction, never at import -- never a Python literal.
`Settings.residency` is validated against that object's `residencies` here; the object itself
(loading, lookups, the deployment self-check) lives in `app.residency`.
"""

from functools import lru_cache

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.jwt_verifier import HS_ALGORITHMS, MIN_HS_SECRET_BYTES, SUPPORTED_JWT_ALGORITHMS
from app.residency import DEFAULT_RESIDENCY_CONFIG_PATH, ResidencyAllowList, ResidencyRoute


class Settings(BaseSettings):
    # secrets_dir (issue #14 / ADR-0011): a secret can come from a file under /run/secrets, one
    # file per field name, instead of (or in addition to) the environment. arbitrary_types_allowed
    # (spec A4 / #110): `residency_allow_list` below is a plain object, not a pydantic model.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        secrets_dir="/run/secrets",
        arbitrary_types_allowed=True,
    )

    app_name: str = "ai-app"
    # No default (issue #14 / ADR-0011): a `.env` copied and left unedited must fail to construct
    # rather than silently becoming a permissive local configuration.
    environment: str
    cors_origins: str = "http://localhost:3000"

    # SecretStr (issue #14 / ADR-0011): repr()/str() never leak a live value. The *pooled* alias
    # only (ADR-0002, Spec 10 / #75); a dedicated tenant's DSN lives under
    # `TENANT_DB_SECRETS_DIR` instead. The owner/migrations DSN is not a field at all here.
    database_url: SecretStr = SecretStr("postgresql+asyncpg://app:app@localhost:5432/app")

    # Explicit pool sizing (Spec 7 / #55): the deployment's real concurrency ceiling (pool_size +
    # max_overflow, per worker) is visible here instead of guessed from driver defaults.
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout: int = 30  # seconds a checkout waits for a free connection
    db_pool_recycle: int = 1800  # seconds before a pooled connection is recycled
    # Per-transaction timeout (ms), set with SET LOCAL in tenant_session().
    db_statement_timeout_ms: int = 30_000

    # A *gateway alias* (docker/litellm/config.yaml's `model_name`), matching the default
    # residency ("eu" only allows "claude-eu"/"embeddings", config/residency.toml, ADR-0009). A
    # `<provider>:<model>` value still works -- `alias_for` strips the prefix.
    llm_model: str = "claude-eu"
    # No default (ADR-0009, ai-app-starter#7 review finding): the gateway is required, not
    # optional -- rejected by `_require_gateway_configured` below, never silently skipped.
    litellm_base_url: str | None = None
    litellm_api_key: SecretStr | None = None
    # The gateway's admin credential (Spec 7 / #53, ADR-0009): mints/revokes per-tenant virtual
    # keys (`app/gateway_provisioning.py`) -- distinct from `litellm_api_key` above.
    litellm_master_key: SecretStr | None = None
    # No default: an unconfigured embedding provider/model must refuse construction rather than
    # silently reaching a default endpoint (ADR-0008).
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimensions: int = 1536
    # Deliberately NOT fields (ai-app-starter#7, ADR-0009): a raw provider key here would be the
    # escape hatch around the gateway. `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` belong to `litellm`
    # alone (docker-compose.yml).

    # The deployment's own residency. Must be a key of `residency_allow_list.residencies` —
    # validated below.
    residency: str = "eu"

    # Where the allow-list is loaded from when `residency_allow_list` below is not supplied
    # directly (e.g. by a test) -- defaults to `app.residency.DEFAULT_RESIDENCY_CONFIG_PATH`.
    residency_config_path: str = str(DEFAULT_RESIDENCY_CONFIG_PATH)

    # The residency allow-list itself (spec A4 / #94, #110), built by `_default_residency_
    # allow_list` below when not supplied directly. `None` only ever marks "not yet built" -- a
    # test passes a `ResidencyAllowList.from_data(...)` instance here instead of a path.
    residency_allow_list: ResidencyAllowList | None = None

    # No default (issue #14 / ADR-0011): a deployment must choose an auth mode explicitly rather
    # than silently running header-based tenant impersonation under what looks like production.
    auth_mode: str = Field(pattern="^(dev-headers|jwt)$")

    langfuse_host: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None

    mcp_tenant_id: str | None = None
    mcp_identity_id: str | None = None

    # MCP transport (issue #48 / ADR-0005): stdio (default, local-dev, identity from
    # MCP_TENANT_ID/MCP_IDENTITY_ID above) or streamable-http (networked, per-connection identity
    # via `app.token_verifier`). `app.mcp.server.check_mcp_mode` mirrors `check_auth_mode` below.
    mcp_transport: str = Field(default="stdio", pattern="^(stdio|streamable-http)$")

    # The deployment's own public Host header(s) for streamable-http (issue #116), comma-
    # separated like `cors_origins` -- feeds the MCP SDK's DNS-rebinding allow-list instead of its
    # 127.0.0.1-only default. No default: `check_mcp_mode` refuses streamable-http without one.
    mcp_allowed_hosts: str = ""

    # The process-wide token issuer for every tenant whose control-plane `identity_issuer` is
    # unset (issue #22 / ADR-0003) -- read by `TenantAuthSettingsRepository.get()` as the
    # fallback default.
    default_identity_issuer: str | None = None

    # JWT verification (issue #24 / ADR-0003 / ADR-0012): the interim single process-wide
    # verification key/algorithm, not a per-tenant JWKS lookup -- a customer-owned identity
    # provider needs a real JWKS key source (change `app.context_resolution.key_source_for`, the
    # one place both the HTTP and MCP adapters read; never `app/jwt_verifier.py`). No
    # default: AUTH_MODE=jwt with no key fails every request as unauthenticated (app/deps.py).
    jwt_verification_key: SecretStr | None = None
    jwt_algorithm: str = "RS256"

    # Agent-credential token exchange (Spec 6 / #47, ADR-0005): mints a short-lived access token
    # when an agent identity exchanges its own credential -- never verifies the tokens of a
    # tenant's identity provider (`jwt_verification_key`/`verify_token`'s job). No default:
    # unconfigured fails every exchange rather than minting an unsigned token.
    #
    # **Deliberately a separate algorithm from `jwt_algorithm` (review finding, Spec 6 /
    # ADR-0005 / ADR-0003)**: a real identity provider signs asymmetrically, this process is both
    # signer and verifier of its own agent tokens, so `agent_token_algorithm` defaults to `HS256`
    # and can be set asymmetric instead (`agent_token_verification_key` below then holds the derived
    # public half). `app.deps.get_key_source`/`get_algorithm_source` pin the matching pair per
    # issuer, never the other one -- the algorithm-confusion guard.
    agent_token_signing_key: SecretStr | None = None
    agent_token_algorithm: str = "HS256"
    # Meaningful only when `agent_token_algorithm` is asymmetric: the public key
    # `app.deps.get_key_source` verifies against. Left unset for the symmetric default; for an
    # asymmetric algorithm with no explicit value, construction below derives it from the private
    # PEM in `agent_token_signing_key` so a deployment only ever manages one secret file.
    agent_token_verification_key: SecretStr | None = None
    # Short-lived by design (ADR-0005): one session's worth of authentication, small blast radius.
    agent_token_ttl_seconds: int = 300

    # Per-membership request limit on the agent-facing routes: a single-process backstop against
    # a stuck client or scripted flood -- NOT the enforcement of record for spend (the gateway's).
    request_limit_max: int = 30
    request_limit_window_seconds: float = 60.0

    # One file per gateway-credential alias (Spec 7 / #52, ADR-0009, ADR-0011), read fresh on
    # every request by `app.gateway_credentials` -- unlike `secrets_dir`, loaded once.
    gateway_credentials_dir: str = "/run/secrets"

    # Default budget/rate limit a newly provisioned tenant's credential is minted with (Spec 7 /
    # #53, ADR-0009) -- tuning defaults, not load-bearing constants.
    gateway_default_spend_ceiling_usd: float = 50.0
    gateway_default_budget_reset_period: str = "30d"
    gateway_default_requests_per_minute: int = 60
    gateway_default_tokens_per_minute: int = 100_000

    # Backup-retention window (Spec 9 / #72, ADR-0010): days backups keep a tenant's last copy
    # after `erase` -- only computes the backup-horizon date an erasure record documents.
    backup_retention_days: int = 30

    @model_validator(mode="after")
    def _require_gateway_configured(self) -> "Settings":
        """Fails closed (ADR-0009, ai-app-starter#7): the gateway is mandatory in every
        environment -- an unset/empty `LITELLM_BASE_URL` refuses construction outright."""
        if not self.litellm_base_url:
            raise ValueError(
                "LITELLM_BASE_URL is required and has no default -- every deployment's compose "
                "stack runs its own LiteLLM gateway (docker-compose.yml's `litellm` service) and "
                "every model/embedding call must go through it, never a provider directly "
                "(ADR-0009). Set it explicitly (e.g. http://litellm:4000 in docker compose; see "
                ".env.example)."
            )
        return self

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
    def _default_residency_allow_list(self) -> "Settings":
        """Builds `residency_allow_list` from `residency_config_path` at construction time, never
        at import (spec A4 / #94, #110), when not supplied directly. Must run before
        `_require_known_residency` below (pydantic v2 runs `mode="after"` validators in
        declaration order)."""
        if self.residency_allow_list is None:
            self.residency_allow_list = ResidencyAllowList.load(self.residency_config_path)
        return self

    @model_validator(mode="after")
    def _require_known_residency(self) -> "Settings":
        assert self.residency_allow_list is not None  # set by the validator above
        if self.residency not in self.residency_allow_list.residencies:
            raise ValueError(
                f"Unknown residency {self.residency!r}. Configured residencies: "
                f"{sorted(self.residency_allow_list.residencies)} (see "
                f"{self.residency_allow_list.source})."
            )
        return self

    @model_validator(mode="after")
    def _require_supported_jwt_algorithms(self) -> "Settings":
        """Algorithm-confusion guard, part 1 (Spec 6 review finding): both algorithm settings
        must name a real signing algorithm -- never "none", never a silent typo."""
        for setting_name, algorithm in (
            ("JWT_ALGORITHM", self.jwt_algorithm),
            ("AGENT_TOKEN_ALGORITHM", self.agent_token_algorithm),
        ):
            if algorithm not in SUPPORTED_JWT_ALGORITHMS:
                raise ValueError(
                    f"{setting_name}={algorithm!r} is not a supported signing algorithm "
                    f"(supported: {sorted(SUPPORTED_JWT_ALGORITHMS)}). 'none' is never accepted, "
                    "for either setting, under any configuration."
                )
        return self

    @model_validator(mode="after")
    def _require_consistent_agent_token_key(self) -> "Settings":
        """Algorithm-confusion guard, part 2 (Spec 6 / ADR-0005): only runs when
        `agent_token_signing_key` is configured. Symmetric (HS*): must be at least
        `MIN_HS_SECRET_BYTES` long. Asymmetric (RS*/ES*/PS*): must be a PEM private key;
        `agent_token_verification_key` is derived from it when unset via
        `app.agent_credential_exchange.derive_public_key_pem` (spec A4 / #94, #112: that module
        owns key derivation) -- this validator only *checks* consistency by calling it."""
        if self.agent_token_signing_key is None:
            return self
        secret_value = self.agent_token_signing_key.get_secret_value()

        if self.agent_token_algorithm in HS_ALGORITHMS:
            if len(secret_value.encode("utf-8")) < MIN_HS_SECRET_BYTES:
                raise ValueError(
                    f"AGENT_TOKEN_SIGNING_KEY is shorter than {MIN_HS_SECRET_BYTES} bytes, too "
                    f"short for AGENT_TOKEN_ALGORITHM={self.agent_token_algorithm!r} -- a short "
                    "HMAC secret makes every agent token forgeable. Use a longer random secret "
                    "(e.g. `openssl rand -hex 32`)."
                )
            return self

        if self.agent_token_verification_key is None:
            # Deferred import: app.agent_credential_exchange imports app.db.session, which
            # imports app.config.get_settings -- a top-level import here would be a real cycle.
            from app.agent_credential_exchange import (
                AgentTokenKeyDerivationError,
                derive_public_key_pem,
            )

            try:
                public_pem = derive_public_key_pem(secret_value)
            except AgentTokenKeyDerivationError as exc:
                raise ValueError(
                    f"AGENT_TOKEN_ALGORITHM={self.agent_token_algorithm!r} is asymmetric: "
                    f"AGENT_TOKEN_SIGNING_KEY {exc}"
                ) from exc
            self.agent_token_verification_key = SecretStr(public_pem)
        return self

    # Run limits (ADR-0009, CONTEXT.md "Run limit") -- never a budget (that's the gateway's).
    run_request_limit: int = 10
    run_tool_calls_limit: int = 20
    run_total_tokens_limit: int | None = None
    run_deadline_seconds: float = 60.0

    # Per-call deadlines (Spec 7 / #54), distinct from `run_deadline_seconds` above (a whole run).
    llm_call_timeout_seconds: float = 30.0
    embedding_call_timeout_seconds: float = 30.0

    # How long a writing-tool approval stays valid (ADR-0007, Spec 5 / #37).
    pending_action_expiry_seconds: float = 300.0

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def mcp_allowed_hosts_list(self) -> list[str]:
        return [h.strip() for h in self.mcp_allowed_hosts.split(",") if h.strip()]

    @property
    def residency_route(self) -> ResidencyRoute:
        """The allow-listed endpoints for this settings object's own residency."""
        assert self.residency_allow_list is not None  # set by _default_residency_allow_list
        return self.residency_allow_list.route_for(self.residency)


@lru_cache
def get_settings() -> Settings:
    return Settings()
