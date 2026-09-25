# Deployment (blueprint A)

## Before exposing anything

- **Auth**: implement `AUTH_MODE=jwt` in `app/deps.py`. The startup guard refuses
  `dev-headers` unless `ENVIRONMENT` is `dev`/`test` — set `ENVIRONMENT=prod` on servers so a
  forgotten auth switch fails loudly instead of running open.
- **Passwords**: set `POSTGRES_PASSWORD` and `APP_DB_PASSWORD` in `.env` (compose interpolates
  them); the defaults are for localhost only.
- **Ports**: compose binds 5432/8000/4000 to `127.0.0.1` — the reverse proxy (below) is the
  only public entry point. Do not "fix" this by unbinding.

## Local / a single server (EU)

- `docker compose up -d` starts Postgres, runs migrations via the one-shot `migrate` service
  (owner role — the api container never holds the superuser DSN), starts the API, and starts
  the model gateway (LiteLLM) — required, not an optional profile (ADR-0009). Set
  `LITELLM_MASTER_KEY` in `.env` first; rendering the compose file fails otherwise.
- Backups: `pg_dump` via cron or provider snapshots; object storage (MinIO/Hetzner) for files.
- Put a reverse proxy with TLS (Caddy/Traefik) in front of the API, targeting `127.0.0.1:8000`.

## Langfuse (tracing)

Langfuse v3 needs ClickHouse, Redis/Valkey, and MinIO. Use the official compose file from
https://github.com/langfuse/langfuse (do not rebuild it), start it on the same Docker network,
and set `LANGFUSE_HOST` (e.g. `http://langfuse-web:3000`), `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY`. Then `uv sync --extra observability`.

## LiteLLM (gateway)

Required (ADR-0009): it runs by default, against its own database and role (`gateway`/`gateway`
in `docker/postgres/01-init.sh`) that can never see the application's tenant tables, and
receives only its own database URL, `LITELLM_MASTER_KEY`, and the model-provider credentials its
configured aliases call — never the application's database credential or the tracing secret.
Maintain `docker/litellm/config.yaml`; set `LITELLM_MASTER_KEY` in `.env` (rendering
`docker-compose.yml` fails if it is unset) and mint virtual keys with per-tenant budgets via the
LiteLLM admin API. Backend: `LITELLM_BASE_URL=http://litellm:4000`,
`LITELLM_API_KEY=<virtual key>`, `LLM_MODEL=openai:claude`, `EMBEDDING_MODEL=embeddings`
(the alias names from the config).

## Scaling / relocation

- More load: scale the API horizontally (it is stateless), run Postgres separately (a managed EU provider).
- Enterprise requirements: move the agent logic to Bedrock AgentCore / Azure Foundry (blueprint C/D);
  the tools remain usable as MCP servers.
- Strict data protection: self-host the models (vLLM) behind the same provider abstraction (blueprint E).
