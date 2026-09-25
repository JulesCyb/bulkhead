# Deployment (blueprint A)

## Before exposing anything

- **Auth**: set `AUTH_MODE=jwt` (implemented in `app/deps.py`, issue #24) and configure
  `JWT_VERIFICATION_KEY`/`JWT_ALGORITHM` plus, per tenant, `control.tenants.identity_issuer` (or
  `DEFAULT_IDENTITY_ISSUER` for the interim one-operator-run-provider case). The startup guard
  refuses `dev-headers` unless `ENVIRONMENT` is `dev`/`test` — set `ENVIRONMENT=prod` on servers
  so a forgotten auth switch fails loudly instead of running open. `JWT_ALGORITHM` is a real IdP's
  own algorithm (default `RS256`; `JWT_VERIFICATION_KEY` is only ever its *public* key) —
  **never** the same setting as `AGENT_TOKEN_ALGORITHM` below, which is this application's own,
  independent, and by default symmetric agent-token algorithm (see `docs/mcp-connection.md` and
  the `AGENT_TOKEN_*` comment block in `.env.example`). If agent identities are in use, also set
  `AGENT_TOKEN_SIGNING_KEY` to a random secret of at least 32 bytes (`openssl rand -hex 32`) — a
  shorter one, or an unsupported/`none` algorithm for either setting, fails `Settings`
  construction outright rather than starting with a weak or forgeable configuration.
- **Passwords**: set `POSTGRES_PASSWORD` and `APP_DB_PASSWORD` in `.env` (compose interpolates
  them); the defaults are for localhost only.
- **Ports**: compose binds 5432/8000/4000 to `127.0.0.1` — the reverse proxy (below) is the
  only public entry point. Do not "fix" this by unbinding.
- **Least privilege by default (#18)**: `api` and `migrate` no longer take `env_file: .env` (a
  blanket copy of the whole file into the container); each names only the variables its own job
  needs in its own `environment:` block. `api` never sees `DATABASE_URL_MIGRATIONS`,
  `APP_OWNER_DB_PASSWORD`, `POSTGRES_PASSWORD`, `GATEWAY_DB_PASSWORD`, or `LITELLM_MASTER_KEY`;
  `migrate` sees only its own owner DSN plus the trace-sink/residency variables it shares with
  `api`. `postgres`/`api`/`migrate` share a `db` network; the gateway is never on it (see
  "LiteLLM (gateway)" below). Adding a new variable to a service means adding it explicitly to
  that service's `environment:` block, not restoring `env_file:`.
- **Secrets as files in production (issue #14 / ADR-0011)**: every secret/connection-string field
  on `app.config.Settings` and `app.migration_settings.MigrationSettings` is read through
  pydantic-settings' `secrets_dir="/run/secrets"` — a file named after the field (e.g.
  `/run/secrets/database_url`, `/run/secrets/anthropic_api_key`) is picked up automatically, and
  a matching environment variable still wins if both are present. In production, mount tenant and
  deployment secrets as files under `/run/secrets` (Docker/Swarm secrets, a Kubernetes
  `secretKeyRef` volume, or a file SOPS decrypts on deploy) instead of passing them as plain
  environment variables in `.env`; rotating a secret is then replacing the file and redeploying,
  with no code change. `.env`/`environment:` stays the right choice for local development.
- **Image pins**: every pulled image in `docker-compose.yml` (`postgres`, `litellm`) names a
  specific version, never `latest`/a bare major-version tag — `docker compose config` is the
  place this is enforced (`tests/test_deployment_hardening_compose.py`,
  `tests/test_gateway_compose.py`). `api`/`migrate` build from this repo's own `Dockerfile`, whose
  own base images (`python:3.12-slim`, `ghcr.io/astral-sh/uv`) are pinned there for the same
  reason.
- **Local env files**: `.gitignore`/`.dockerignore` exclude any `.env.<anything>` (not just the
  exact name `.env`) by default, so a locally created `.env.local`/`.env.production` never lands
  in git or an image layer; `.env.example` is explicitly re-included in both so the template stays
  tracked and shipped.

## Local / a single server (EU)

- `docker compose up -d` starts Postgres, runs migrations via the one-shot `migrate` service
  (as the non-superuser `app_owner` role — the `api` container's own environment allow-list
  never includes `DATABASE_URL_MIGRATIONS`, `APP_OWNER_DB_PASSWORD`, or `POSTGRES_PASSWORD`; see
  "Least privilege by default" above), starts the API, and starts the model gateway (LiteLLM) —
  required, not an optional profile (ADR-0009). Set `LITELLM_MASTER_KEY` in `.env` first;
  rendering the compose file fails otherwise.
- Backups: `pg_dump` via cron or provider snapshots; object storage (MinIO/Hetzner) for files.
- Put a reverse proxy with TLS (Caddy/Traefik) in front of the API, targeting `127.0.0.1:8000`.

## Residency (ADR-0008)

Every host a content-bearing path can reach — the database, the model gateway, and the trace
sink — must be inside the deployment's residency, the same requirement for all three: a EU
deployment's `DATABASE_URL`/gateway host and its `LANGFUSE_HOST` must both resolve inside the EU,
just as a US deployment's must both resolve inside the US. `RESIDENCY` in `.env` names which one
this deployment is; `RESIDENCY_ALLOW_LIST` (loaded by `app/config.py` from
[`config/residency.toml`](../config/residency.toml), path overridable with
`RESIDENCY_CONFIG_PATH`) is the single place that lists each residency's allowed hosts — add a
residency there (a new `[residency.<name>]` table, see `docs/residency.md`), not by editing a host
string in one of these sections. Malformed or missing, the process refuses to start.

`api` and `migrate` each name `LANGFUSE_HOST` (and `RESIDENCY`) explicitly in their own
`environment:` block in `docker-compose.yml` (#18 dropped the blanket `env_file: .env` both used
to have); `postgres` and `litellm` receive only the specific variables named under their own
`environment:` block and never see the tracing secret (see `tests/test_gateway_compose.py`).
`tests/test_residency_trace_sink_compose.py` renders the compose file and asserts this wiring
directly, so an edit that silently drops the trace sink from a service that needs it — or leaks it
into one that shouldn't have it — fails that test instead of surfacing in an incident.

## Langfuse (tracing)

Langfuse v3 needs ClickHouse, Redis/Valkey, and MinIO. Use the official compose file from
https://github.com/langfuse/langfuse (do not rebuild it), start it on the same Docker network,
inside the deployment's residency (see above), and set `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY` (ADR-0008: the trace sink host itself is resolved per tenant from
`RESIDENCY_ALLOW_LIST`, not from `LANGFUSE_HOST` — see `app/observability.py`). The tracing
dependency (`logfire`, plus the OpenTelemetry SDK/OTLP exporter) is a regular dependency, always
installed — no extra `uv sync` step. Traces carry identifiers only; a tenant admin opts their own
tenant into content capture via `tenants.settings["content_tracing_opt_in"]`
(`app/tenant_settings.py`).

## LiteLLM (gateway)

Required (ADR-0009): it runs by default, against its own database and role (`gateway`/`gateway`
in `docker/postgres/01-init.sh`) that can never see the application's tenant tables, and
receives only its own database URL, `LITELLM_MASTER_KEY`, and the model-provider credentials its
configured aliases call — never the application's database credential or the tracing secret.
Maintain `docker/litellm/config.yaml`; set `LITELLM_MASTER_KEY` in `.env` (rendering
`docker-compose.yml` fails if it is unset). The operator tool's `create` command
(`scripts/operator.py create ...`, Spec 9 / #70) mints every tenant's virtual key automatically,
using the same credential-issuance primitives as
`app.gateway_provisioning.provision_gateway_credential` (Spec 7 / #53) — no manual step.
`revoke_gateway_credential(tenant_id)` reverses provisioning outside of `create` (revokes the
key, removes the secret file, clears the control-plane alias); a future `erase` command
(Spec 9) calls it as part of removing a tenant. Backend: `LITELLM_BASE_URL=http://litellm:4000`,
`LITELLM_API_KEY=<virtual key>`, `LLM_MODEL=openai:claude`, `EMBEDDING_MODEL=embeddings`
(the alias names from the config).

**Networking (#18)**: `litellm` sits on two networks, neither shared with `db` (the network `api`
and `migrate` use for the application's own connections to postgres): `gatewaydb` carries only the
gateway's own database traffic to `postgres`, and `gateway` carries only `api`'s HTTP calls to the
gateway's OpenAI-compatible API. A compromised gateway credential therefore has no network route
to the application's own database traffic, and `migrate` (which never calls the gateway) has no
route to it at all. `postgres` and `litellm` still share the `gatewaydb` network — the gateway's
own database genuinely lives on that same Postgres instance (ADR-0009), and a Postgres bound to
`127.0.0.1` on the host cannot be reached from a container with no shared Docker network at all;
splitting the gateway onto its own Postgres instance would remove that overlap entirely but is a
bigger change than this networking pass (see `tests/test_deployment_hardening_compose.py`).

## Scaling / relocation

- More load: scale the API horizontally (it is stateless), run Postgres separately (a managed EU provider).
- Enterprise requirements: move the agent logic to Bedrock AgentCore / Azure Foundry (blueprint C/D);
  the tools remain usable as MCP servers.
- Strict data protection: self-host the models (vLLM) behind the same provider abstraction (blueprint E).
