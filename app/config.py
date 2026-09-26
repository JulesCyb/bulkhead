"""Central configuration. Values come from the environment or .env (see .env.example).

Tenant-specific things (model choice, prompts, limits) do NOT belong here — they live in
tenants.settings. This holds process-wide settings only.

Residency allow-list (ADR-0008, spec A4 / #94, #110): `Settings.residency_allow_list` holds one
`app.residency.ResidencyAllowList` instance -- for each residency (jurisdiction), the host
patterns and endpoints its content-bearing calls may reach and the gateway model aliases it may
use. Built, by default, from `config/residency.toml` (path overridable with the
`RESIDENCY_CONFIG_PATH` environment variable) at `Settings` construction, never at import --
never a Python literal to edit, so a deployment can change or extend its allow-list without a
code change, and a config-management tool can template the file directly. Adding a residency is a
new `[residency.<name>]` table in that file, not a code change scattered across the
model-routing, embeddings, and observability modules. `Settings.residency` (the deployment's own
residency, or a tenant's `control.tenants.residency` at the call sites that resolve it) is
validated against that object's `residencies`, so an unknown residency identifier is rejected
here, not discovered later at request time. This module only holds the field and its
known-residency validator; the object itself (loading, validation, lookups, and the deployment
self-check) lives in `app.residency`, and the startup walk and per-tenant resolver that consume it
are `app.startup_checks.run_startup_checks` and `app.residency.resolve_residency_route`.
"""

from functools import lru_cache

from cryptography.hazmat.primitives import serialization
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.residency import DEFAULT_RESIDENCY_CONFIG_PATH, ResidencyAllowList, ResidencyRoute

# Role-level settings for the `app` role (Spec 7 / #55): applied once in
# docker/postgres/01-init.sh, mirrored here so the embedded-Postgres integration test can assert
# them without duplicating literals. Independent of Settings.db_statement_timeout_ms below, which
# is the per-transaction timeout the application sets on every tenant_session().
ROLE_STATEMENT_TIMEOUT_MS = 60_000
ROLE_CONNECTION_LIMIT = 50


# Algorithm-confusion guard, shared by `jwt_algorithm` and `agent_token_algorithm` below (Spec 6
# review finding: human and agent tokens must never be checkable with each other's algorithm or
# key). Every algorithm PyJWT's `cryptography` backend supports for signing -- deliberately never
# includes "none" or an empty string, so `Settings` construction itself is the first place a
# downgrade-to-unsigned configuration is refused, before `app.jwt_verifier.verify_token` (which
# also never accepts "none" -- it always passes an explicit `algorithms` allow-list to
# `jwt.decode`) ever sees a token.
_SUPPORTED_JWT_ALGORITHMS = frozenset(
    {
        "HS256",
        "HS384",
        "HS512",
        "RS256",
        "RS384",
        "RS512",
        "ES256",
        "ES384",
        "ES512",
        "PS256",
        "PS384",
        "PS512",
    }
)
_HS_ALGORITHMS = frozenset({"HS256", "HS384", "HS512"})
# NIST SP 800-107 / RFC 2104: an HMAC key shorter than its hash's output size is weaker than the
# hash offers -- 32 bytes is the floor for every HS* algorithm above (HS256's own digest size),
# so one constant covers all three rather than sizing per algorithm.
MIN_HS_SECRET_BYTES = 32


class Settings(BaseSettings):
    # secrets_dir (issue #14 / ADR-0011): in production, tenant secrets and connection strings can
    # be supplied as files under /run/secrets (one file per field name) instead of, or in addition
    # to, the process environment — a value here is still overridden by the matching environment
    # variable if both are present. This lets an operator rotate a secret by replacing a file and
    # redeploying, with no code change.
    # arbitrary_types_allowed (spec A4 / #110): `residency_allow_list` below holds a plain
    # `ResidencyAllowList` instance, not a pydantic model -- pydantic must be told it is allowed
    # as a field type rather than something to validate/coerce.
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
    # privileges. Alembic's own environment module and the operator tool (app/operator/cli.py)
    # resolve it from app.migration_settings instead — a source app.main and app.deps never
    # import.

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

    # Default is a *gateway alias* (docker/litellm/config.yaml's `model_name`), not a
    # "<provider>:<model>" id -- it must match the deployment's own default residency ("eu" here)
    # so a fresh deployment's default configuration passes `app.startup_checks.run_startup_checks`
    # unmodified (ai-app-starter#7 review finding): residency "eu"'s model allow-list only allows
    # "claude-eu"/"embeddings" (config/residency.toml, ADR-0009), never a raw provider id such as
    # "anthropic:claude-sonnet-4-5" (that used to be this field's default, back when the gateway
    # was optional and the model check only ran once one was configured). A `<provider>:<model>`
    # value still works too -- `validate_model_for_residency`/`_bare_model_name` strip any
    # "<provider>:" prefix before checking -- but the gateway's own bare alias is the common case.
    llm_model: str = "claude-eu"
    # No default (ADR-0009, ai-app-starter#7 review finding): the LiteLLM gateway is a required
    # service, not an optional profile -- every deployment's compose stack runs it, and every
    # model/embedding call must go through it, never a provider directly. Typed `str | None` only
    # so an explicitly empty/unset value can be told apart from a real URL and rejected with a
    # clear message (`_require_gateway_configured` below) instead of failing on the wrong field.
    # Nothing in this codebase may construct a provider client without first reading this value --
    # `app.llm`, the only module that builds a chat client, has no code path around it.
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
    # Deliberately NOT fields here (ai-app-starter#7 review finding, ADR-0009): a raw provider API
    # key on this object would be exactly the escape hatch that lets application code build a
    # client that skips the gateway. `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` are real environment
    # variables in `.env`/compose, but they belong to the `litellm` gateway service alone
    # (`docker-compose.yml`'s `litellm:` service, `docker/litellm/config.yaml`) -- the long-running
    # API process this class configures never receives them and has no field to hold them in.

    # The deployment's (or, at a call site resolving a tenant's own setting, that tenant's)
    # residency. Must be a key of `residency_allow_list.residencies` — validated below.
    residency: str = "eu"

    # Where the residency allow-list is loaded from, if `residency_allow_list` below is not
    # supplied directly (e.g. by a test) -- the `RESIDENCY_CONFIG_PATH` environment variable,
    # defaulting to `app.residency.DEFAULT_RESIDENCY_CONFIG_PATH`. Kept as its own field (rather
    # than folded silently into the loader) so an operator or a diagnostic endpoint can report
    # which file a running process actually loaded.
    residency_config_path: str = str(DEFAULT_RESIDENCY_CONFIG_PATH)

    # The residency allow-list itself (spec A4 / #94, #110): one `ResidencyAllowList` instance,
    # built from `residency_config_path` at construction time by `_default_residency_allow_list`
    # below when not supplied directly. `None` here is never a real value once construction
    # finishes -- it only marks "not yet built"; every reader (this class's own
    # `_require_known_residency`/`residency_route`, `app.residency.resolve_residency_route`,
    # `app.startup_checks.run_startup_checks`, and the other call sites named in `app.residency`'s
    # own module docstring) sees a real `ResidencyAllowList` there. A test builds one directly
    # (`ResidencyAllowList.from_data(...)`) and passes it here instead of pointing
    # `residency_config_path` at a file on disk.
    residency_allow_list: ResidencyAllowList | None = None

    # No default (issue #14 / ADR-0011): same reasoning as `environment` above — a deployment
    # must choose an auth mode explicitly rather than silently running header-based tenant
    # impersonation under what looks like production.
    auth_mode: str = Field(pattern="^(dev-headers|jwt)$")

    langfuse_host: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None

    mcp_tenant_id: str | None = None
    mcp_identity_id: str | None = None

    # MCP transport (issue #48 / ADR-0005): a single documented setting choosing whether the
    # tools server speaks its local-development transport (stdio, the process-wide identity from
    # MCP_TENANT_ID/MCP_IDENTITY_ID above) or the networked one (streamable-http, a per-connection
    # identity derived from a verified token via app.token_verifier). stdio is the default, so
    # moving from a laptop to a real deployment is a configuration change -- one
    # `app.mcp.server.check_mcp_mode` refuses to leave half-finished (ADR-0005's guard, mirroring
    # `check_auth_mode` above).
    mcp_transport: str = Field(default="stdio", pattern="^(stdio|streamable-http)$")

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

    # Agent-credential token exchange (Spec 6 / #47, ADR-0005): the signing counterpart to
    # jwt_verification_key above -- used only to mint a short-lived access token when an agent
    # identity exchanges its own credential, never to verify a customer-owned identity provider's
    # tokens (that stays jwt_verification_key/verify_token's job). Kept as its own SecretStr field
    # (file-backed via secrets_dir, same as every other secret here) so a deployment can rotate it
    # independently of jwt_verification_key. No default: an unconfigured signing key fails every
    # exchange rather than silently minting an unsigned or otherwise weak token.
    #
    # **Deliberately a separate algorithm from `jwt_algorithm` above (review finding, Spec 6 /
    # ADR-0005 / ADR-0003).** A real tenant IdP signs human tokens with an asymmetric algorithm
    # (RS256/ES256 -- `jwt_verification_key` holds only its *public* key, never a secret this
    # deployment could forge a token with); the tokens this application mints for its own agent
    # identities are naturally symmetric (there is no "IdP" on the other end to keep a private key
    # from -- this process is both signer and verifier). Sharing one `jwt_algorithm` setting
    # between the two, as an earlier version of this starter did, meant the first realistic
    # production config (`JWT_ALGORITHM=RS256` for the real IdP) silently broke every agent token,
    # since minting and verifying an agent token both used that same, now-asymmetric, algorithm
    # against a plain shared secret. `agent_token_algorithm` defaults to `HS256` -- the common
    # case, where `agent_token_signing_key` alone is both the signing and verification secret --
    # and can be set to an asymmetric algorithm instead, in which case
    # `agent_token_verification_key` below (explicit, or derived from this field, see there) holds
    # the public half. The issuer-aware key/algorithm pinning in `app.deps.get_key_source` /
    # `get_algorithm_source` (used by both the HTTP API and the MCP transport, via
    # `app.token_verifier.verify_tenant_token`) is what actually enforces, per token, that a human
    # (tenant-IdP) issuer is only ever checked against `jwt_verification_key`/`jwt_algorithm` and
    # an agent issuer only ever against `agent_token_signing_key` (or
    # `agent_token_verification_key`)/`agent_token_algorithm` -- never the other pair. This is the
    # algorithm-confusion guard: neither `app.jwt_verifier.verify_token` nor `jwt.decode` itself
    # is ever told to accept an algorithm the token's own header names (`algorithms` is always an
    # explicit allow-list this settings object resolved ahead of time), so a token cannot pick its
    # own verification algorithm, and "none" is never in `_SUPPORTED_JWT_ALGORITHMS` at all.
    agent_token_signing_key: SecretStr | None = None
    agent_token_algorithm: str = "HS256"
    # Only meaningful when `agent_token_algorithm` is asymmetric (see field docstring above): the
    # public key `app.deps.get_key_source` verifies an agent token against, paired with the
    # private key in `agent_token_signing_key`. Left unset for the default symmetric case (nothing
    # to derive -- `agent_token_signing_key` alone serves both roles), and for an asymmetric
    # algorithm with no explicit value here, `Settings` construction below derives it from
    # `agent_token_signing_key` (which must then be a PEM private key) so a deployment only ever
    # has to manage one secret file. Set it explicitly instead when the private key is not this
    # process's to hold at all (e.g. minted by a separate signer).
    agent_token_verification_key: SecretStr | None = None
    # Short-lived by design (ADR-0005): long enough for one connection/tool-call session to
    # authenticate once, short enough that a leaked token has a small blast radius.
    agent_token_ttl_seconds: int = 300

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

    # Backup-retention window (Spec 9 / #72, ADR-0010): how long, in days, the deployment's own
    # backup system keeps the last copy of a tenant's data after the operator tool's `erase`
    # command removes it everywhere else. Used only to compute the backup-horizon date written
    # onto each erasure record -- the tool documents this date, it never acts on it (no automatic
    # purge), so nobody tells a tenant "your data is gone" while a backup still holds a copy.
    backup_retention_days: int = 30

    @model_validator(mode="after")
    def _require_gateway_configured(self) -> "Settings":
        """Fails closed (ADR-0009, ai-app-starter#7 review finding): the LiteLLM gateway is
        mandatory in every environment -- the compose stack always runs it (`docker-compose.yml`'s
        `litellm:` service) -- so an unset or empty `LITELLM_BASE_URL` must refuse to construct a
        `Settings` object at all, not just skip a check further downstream. This is what makes
        `app.startup_checks.run_startup_checks`'s own gateway-host/residency check unconditional:
        by the time that function runs, `settings.litellm_base_url` is always truthy. Do not
        relax this to `str | None` being an accepted "no gateway" profile again -- that was the
        exact finding (ADR-0009, Spec 7 / #57/#63 follow-up) this validator closes."""
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
        """Builds `residency_allow_list` from `residency_config_path` at construction time --
        never at import (spec A4 / #94, #110) -- when the caller has not supplied one directly.
        Must run before `_require_known_residency` below, which validates `residency` against it;
        pydantic v2 runs `model_validator(mode="after")` hooks in the order they are defined on
        the class, so this one is placed immediately above that one."""
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
        """Algorithm-confusion guard, part 1 (review finding, Spec 6): both algorithm settings
        must name a real signing algorithm this deployment actually intends -- never "none", never
        a typo that `jwt.decode`'s own `algorithms` allow-list would otherwise just silently never
        match (which fails safe, but with a confusing "invalid or expired" 401 instead of a
        startup error naming the actual mistake)."""
        for setting_name, algorithm in (
            ("JWT_ALGORITHM", self.jwt_algorithm),
            ("AGENT_TOKEN_ALGORITHM", self.agent_token_algorithm),
        ):
            if algorithm not in _SUPPORTED_JWT_ALGORITHMS:
                raise ValueError(
                    f"{setting_name}={algorithm!r} is not a supported signing algorithm "
                    f"(supported: {sorted(_SUPPORTED_JWT_ALGORITHMS)}). 'none' is never accepted, "
                    "for either setting, under any configuration."
                )
        return self

    @model_validator(mode="after")
    def _require_consistent_agent_token_key(self) -> "Settings":
        """Algorithm-confusion guard, part 2, and the fail-closed startup check for the agent
        signing key (review finding, Spec 6 / ADR-0005): nothing here runs unless
        `agent_token_signing_key` is actually configured -- an unconfigured key already fails
        every exchange at request time (`app/agent_credential_exchange.py`) by design, and this
        validator's job is only to refuse a key that *is* configured but too weak, or
        asymmetric-but-incomplete, before the process ever mints or verifies a single token.

        - `agent_token_algorithm` is symmetric (HS*): `agent_token_signing_key` doubles as the
          verification secret (`app.deps.get_key_source`) -- it must be at least
          `MIN_HS_SECRET_BYTES` (32) bytes, the same floor RFC 2104 / NIST SP 800-107 recommend for
          an HMAC key at least as long as the hash's own output, or a short, guessable secret would
          make every agent token forgeable.
        - `agent_token_algorithm` is asymmetric (RS*/ES*/PS*): `agent_token_signing_key` must be a
          PEM-encoded private key (minting needs it), and `agent_token_verification_key` (the
          public half `get_key_source` checks incoming tokens against) is derived from it
          automatically when not set explicitly -- so a deployment switching to an asymmetric
          agent-token algorithm only ever has to manage the one private-key secret, and a
          malformed private key is refused here rather than at the first mint/verify call.
        """
        if self.agent_token_signing_key is None:
            return self
        secret_value = self.agent_token_signing_key.get_secret_value()

        if self.agent_token_algorithm in _HS_ALGORITHMS:
            if len(secret_value.encode("utf-8")) < MIN_HS_SECRET_BYTES:
                raise ValueError(
                    f"AGENT_TOKEN_SIGNING_KEY is shorter than {MIN_HS_SECRET_BYTES} bytes, too "
                    f"short for AGENT_TOKEN_ALGORITHM={self.agent_token_algorithm!r} -- a short "
                    "HMAC secret makes every agent token forgeable. Use a longer random secret "
                    "(e.g. `openssl rand -hex 32`)."
                )
            return self

        # Asymmetric agent_token_algorithm: agent_token_signing_key must be the PEM private key;
        # derive the matching public key for verification when none was set explicitly.
        if self.agent_token_verification_key is None:
            try:
                private_key = serialization.load_pem_private_key(
                    secret_value.encode("utf-8"), password=None
                )
                public_pem = (
                    private_key.public_key()
                    .public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo,
                    )
                    .decode("utf-8")
                )
            except Exception as exc:
                raise ValueError(
                    f"AGENT_TOKEN_ALGORITHM={self.agent_token_algorithm!r} is asymmetric: "
                    "AGENT_TOKEN_SIGNING_KEY must be a PEM-encoded private key so its public key "
                    "can be derived for verification, or set AGENT_TOKEN_VERIFICATION_KEY "
                    "explicitly to the matching public key."
                ) from exc
            self.agent_token_verification_key = SecretStr(public_pem)
        return self

    # Run limits (ADR-0009, CONTEXT.md "Run limit"): the ceiling of model requests, tool calls,
    # and wall-clock time a single agent run may consume, enforced in the application — never a
    # budget, that lives at the gateway. "Order of ten model requests and twenty tool calls" is a
    # starting default meant to be tuned per deployment, not a load-bearing constant.
    run_request_limit: int = 10
    run_tool_calls_limit: int = 20
    run_total_tokens_limit: int | None = None
    run_deadline_seconds: float = 60.0

    # Per-call wall-clock deadlines (Spec 7 / #54, ADR-0009): distinct from `run_deadline_seconds`
    # above, which bounds a whole agent run (many model requests and tool calls). These bound one
    # single model or embedding request, replacing the client library's own multi-minute default,
    # and are sized under the reverse proxy's own timeout.
    llm_call_timeout_seconds: float = 30.0
    embedding_call_timeout_seconds: float = 30.0

    # Pending-action approval window (ADR-0007, Spec 5 / #37): how long a writing-tool approval
    # request stays valid before it can no longer be approved -- configuration, not a constant,
    # so a deployment can tune it to how quickly its members typically respond.
    pending_action_expiry_seconds: float = 300.0

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def residency_route(self) -> ResidencyRoute:
        """The allow-listed endpoints for this settings object's own residency."""
        assert self.residency_allow_list is not None  # set by _default_residency_allow_list
        return self.residency_allow_list.route_for(self.residency)


@lru_cache
def get_settings() -> Settings:
    return Settings()
