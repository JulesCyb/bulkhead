"""Central configuration. Values come from the environment or .env (see .env.example).

Tenant-specific things (model choice, prompts, limits) do NOT belong here — they live in
tenants.settings. This holds process-wide settings only.

Residency allow-list (ADR-0008, Spec 8 / #63): ``RESIDENCY_ALLOW_LIST`` and
``RESIDENCY_MODEL_ALLOW_LIST`` below name, for each residency (jurisdiction), the host patterns
and endpoints its content-bearing calls may reach and the gateway model aliases it may use. This
is configuration data loaded once at import time from ``config/residency.toml`` (path overridable
with the ``RESIDENCY_CONFIG_PATH`` environment variable, see ``load_residency_config`` below) --
never a Python literal to edit, so a deployment can change or extend its allow-list without a
code change, and a config-management tool can template the file directly. Adding a residency is a
new ``[residency.<name>]`` table in that file, not a code change scattered across the
model-routing, embeddings, and observability modules. ``Settings.residency`` (the deployment's own
residency, or a tenant's ``control.tenants.residency`` at the call sites that resolve it) is
validated against this allow-list's keys, so an unknown residency identifier is rejected here, not
discovered later at request time. This module only loads/validates the allow-list; the startup
walk that checks every configured endpoint against its residency's allow-list, and the resolver
that returns a route for a given residency, are built in the startup-validation and
residency-resolution modules that consume this data.
"""

import tomllib
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Role-level settings for the `app` role (Spec 7 / #55): applied once in
# docker/postgres/01-init.sh, mirrored here so the embedded-Postgres integration test can assert
# them without duplicating literals. Independent of Settings.db_statement_timeout_ms below, which
# is the per-transaction timeout the application sets on every tenant_session().
ROLE_STATEMENT_TIMEOUT_MS = 60_000
ROLE_CONNECTION_LIMIT = 50

# Default location of the residency allow-list file, relative to the repository root (this file
# lives at <repo>/app/config.py). ``RESIDENCY_CONFIG_PATH`` overrides it -- read directly from the
# environment here, not through `Settings`, because the allow-list must exist as a module-level
# constant before any `Settings` instance can validate its own `residency` field against it.
DEFAULT_RESIDENCY_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "residency.toml"


class _ResidencyConfigLocation(BaseSettings):
    """Reads only `RESIDENCY_CONFIG_PATH`, from the same `.env` file (and `/run/secrets`) every
    other setting uses -- a separate, minimal `BaseSettings` (mirrors `app.migration_settings`'s
    pattern) because the allow-list itself must be loaded and validated before the main
    `Settings` class exists, so `Settings._require_known_residency` has something to validate
    `residency` against."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", secrets_dir="/run/secrets"
    )

    residency_config_path: str = str(DEFAULT_RESIDENCY_CONFIG_PATH)


class ResidencyRoute(BaseModel):
    """The endpoints one residency may reach on a content-bearing path."""

    model_host_patterns: tuple[str, ...]
    embedding_endpoint: str
    trace_sink_host: str


class ResidencyConfigError(ValueError):
    """The residency allow-list configuration file is missing, unreadable, malformed, empty, or
    lets two residencies reach the same host. Fails closed and always names the offending file
    and the exact problem, so an operator (or an AI agent extending this template) can fix it
    without reading this module's source. Raised at import time -- a broken allow-list must stop
    the process from ever starting, never be discovered later at request time."""


def load_residency_config(
    path: str | Path | None = None,
) -> tuple[dict[str, ResidencyRoute], dict[str, tuple[str, ...]]]:
    """Loads and validates the residency allow-list from a TOML file (ADR-0008, Spec 8 / #63).

    `path` defaults to the `RESIDENCY_CONFIG_PATH` environment variable, then to
    `DEFAULT_RESIDENCY_CONFIG_PATH`. Each `[residency.<name>]` table must define
    `model_host_patterns` (a non-empty list of glob host patterns), `embedding_endpoint`, and
    `trace_sink_host` (both non-empty strings); `models` (a list of gateway alias names) is
    optional and becomes that residency's entry in the returned model allow-list.

    Every model/embedding call goes through the LiteLLM gateway (ADR-0009), so
    `model_host_patterns` and `embedding_endpoint` must name *that residency's own* gateway
    host(s) -- never a raw, globally-reachable provider domain such as `*.anthropic.com` or
    `*.openai.com`, which is reachable from every jurisdiction and therefore enforces nothing.
    This function actively rejects a file in which two residencies name the same host or pattern
    (in `model_host_patterns`, `embedding_endpoint`'s host, or `trace_sink_host`): residencies
    that can reach the same host are not actually separated, whatever their allow-lists claim.

    Raises `ResidencyConfigError` for every failure mode above -- never a partial or best-effort
    allow-list, and never a silent fallback to an empty or default configuration.
    """
    config_path = (
        Path(path) if path is not None else Path(_ResidencyConfigLocation().residency_config_path)
    )
    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ResidencyConfigError(
            f"residency configuration file not found or unreadable: {config_path} ({exc}). "
            "Set RESIDENCY_CONFIG_PATH, or restore config/residency.toml (see docs/residency.md)."
        ) from exc
    try:
        data = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as exc:
        raise ResidencyConfigError(
            f"residency configuration file {config_path} is not valid TOML: {exc}"
        ) from exc

    residencies = data.get("residency")
    if not isinstance(residencies, dict) or not residencies:
        raise ResidencyConfigError(
            f"residency configuration file {config_path} must define at least one "
            "[residency.<name>] table"
        )

    allow_list: dict[str, ResidencyRoute] = {}
    model_allow_list: dict[str, tuple[str, ...]] = {}
    for name, entry in residencies.items():
        if not isinstance(entry, dict):
            raise ResidencyConfigError(
                f"{config_path}: [residency.{name}] must be a table, got {type(entry).__name__}"
            )
        required_keys = ("model_host_patterns", "embedding_endpoint", "trace_sink_host")
        missing = [k for k in required_keys if k not in entry]
        if missing:
            raise ResidencyConfigError(
                f"{config_path}: [residency.{name}] is missing required key(s): {missing}"
            )
        host_patterns = entry["model_host_patterns"]
        if (
            not isinstance(host_patterns, list)
            or not host_patterns
            or not all(isinstance(p, str) and p for p in host_patterns)
        ):
            raise ResidencyConfigError(
                f"{config_path}: [residency.{name}].model_host_patterns must be a non-empty "
                "list of non-empty strings"
            )
        embedding_endpoint = entry["embedding_endpoint"]
        trace_sink_host = entry["trace_sink_host"]
        if not isinstance(embedding_endpoint, str) or not embedding_endpoint:
            raise ResidencyConfigError(
                f"{config_path}: [residency.{name}].embedding_endpoint must be a non-empty string"
            )
        if not isinstance(trace_sink_host, str) or not trace_sink_host:
            raise ResidencyConfigError(
                f"{config_path}: [residency.{name}].trace_sink_host must be a non-empty string"
            )
        models = entry.get("models", [])
        if not isinstance(models, list) or not all(isinstance(m, str) and m for m in models):
            raise ResidencyConfigError(
                f"{config_path}: [residency.{name}].models must be a list of non-empty strings"
            )

        allow_list[name] = ResidencyRoute(
            model_host_patterns=tuple(host_patterns),
            embedding_endpoint=embedding_endpoint,
            trace_sink_host=trace_sink_host,
        )
        if models:
            model_allow_list[name] = tuple(models)

    # Fail closed if two residencies could ever reach the same host: that defeats the entire
    # point of a per-residency allow-list. Compared as literal strings (a glob pattern is only
    # ever equal to itself here), which is enough to catch the copy-paste mistake this fix
    # exists for -- two residencies naming the exact same provider host.
    host_owner: dict[str, str] = {}
    for name, route in allow_list.items():
        embedding_host = route.embedding_endpoint.split("//", 1)[-1].split("/", 1)[0]
        for host in (*route.model_host_patterns, embedding_host, route.trace_sink_host):
            owner = host_owner.get(host)
            if owner is not None and owner != name:
                raise ResidencyConfigError(
                    f"{config_path}: host/pattern {host!r} is allow-listed for both "
                    f"{owner!r} and {name!r} -- residencies must never share a host"
                )
            host_owner[host] = name

    return allow_list, model_allow_list


# Loaded once at import time (ADR-0008, Spec 8 / #63): a broken or misconfigured allow-list file
# must stop the process before it ever accepts a request, not be discovered when the first tenant
# request tries to resolve a route.
RESIDENCY_ALLOW_LIST: dict[str, ResidencyRoute]
RESIDENCY_MODEL_ALLOW_LIST: dict[str, tuple[str, ...]]
RESIDENCY_ALLOW_LIST, RESIDENCY_MODEL_ALLOW_LIST = load_residency_config()


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

    # Where the residency allow-list (`RESIDENCY_ALLOW_LIST`/`RESIDENCY_MODEL_ALLOW_LIST` above)
    # is loaded from -- informational only on this object (the allow-list itself is loaded once
    # at import time, before any `Settings` instance exists, via the `RESIDENCY_CONFIG_PATH`
    # environment variable read directly in `load_residency_config`); kept here too so an operator
    # or a diagnostic endpoint can report which file a running process actually loaded.
    residency_config_path: str = str(DEFAULT_RESIDENCY_CONFIG_PATH)

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
    # tokens (that stays jwt_verification_key/verify_token's job). For the symmetric algorithm
    # this starter defaults verification to, this is literally the same secret as
    # jwt_verification_key; kept as its own SecretStr field (file-backed via secrets_dir, same as
    # every other secret here) so a deployment can rotate or split it independently. No default:
    # an unconfigured signing key fails every exchange rather than silently minting an unsigned
    # or otherwise weak token.
    agent_token_signing_key: SecretStr | None = None
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
        return RESIDENCY_ALLOW_LIST[self.residency]


@lru_cache
def get_settings() -> Settings:
    return Settings()
