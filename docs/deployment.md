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

## Residency (ADR-0008)

Every host a content-bearing path can reach — the database, the model gateway, and the trace
sink — must be inside the deployment's residency, the same requirement for all three: a EU
deployment's `DATABASE_URL`/gateway host and its `LANGFUSE_HOST` must both resolve inside the EU,
just as a US deployment's must both resolve inside the US. `RESIDENCY` in `.env` names which one
this deployment is; `RESIDENCY_ALLOW_LIST` in `app/config.py` is the single place that lists each
residency's allowed hosts — add a residency there, not by editing a host string in one of these
sections.

`api` and `migrate` are the two services with `env_file: .env` in `docker-compose.yml`, so both
receive `LANGFUSE_HOST` (and `RESIDENCY`) exactly as set in `.env`; `postgres` and `litellm`
receive only the specific variables named under their own `environment:` block and never see the
tracing secret (see `tests/test_gateway_compose.py`). `tests/test_residency_trace_sink_compose.py`
renders the compose file and asserts this wiring directly, so an edit that silently drops the
trace sink from a service that needs it — or leaks it into one that shouldn't have it — fails
that test instead of surfacing in an incident.

## Langfuse (tracing)

Langfuse v3 needs ClickHouse, Redis/Valkey, and MinIO. Use the official compose file from
https://github.com/langfuse/langfuse (do not rebuild it), start it on the same Docker network,
inside the deployment's residency (see above), and set `LANGFUSE_HOST`
(e.g. `http://langfuse-web:3000`), `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`. Then
`uv sync --extra observability`.

## LiteLLM (gateway)

Required (ADR-0009): it runs by default, against its own database and role (`gateway`/`gateway`
in `docker/postgres/01-init.sh`) that can never see the application's tenant tables, and
receives only its own database URL, `LITELLM_MASTER_KEY`, and the model-provider credentials its
configured aliases call — never the application's database credential or the tracing secret.
Maintain `docker/litellm/config.yaml`; set `LITELLM_MASTER_KEY` in `.env` (rendering
`docker-compose.yml` fails if it is unset). `scripts/seed.py` mints the first tenant's virtual
key automatically via `app.gateway_provisioning.provision_gateway_credential` (Spec 7 / #53) —
no manual step. Before Spec 9's operator tool exists, provision any later tenant the same way,
from a Python shell: `await provision_gateway_credential(tenant_id, residency=..., limits=...)`;
`revoke_gateway_credential(tenant_id)` reverses it (revokes the key, removes the secret file,
clears the control-plane alias). Backend: `LITELLM_BASE_URL=http://litellm:4000`,
`LITELLM_API_KEY=<virtual key>`, `LLM_MODEL=openai:claude`, `EMBEDDING_MODEL=embeddings`
(the alias names from the config).

## Scaling / relocation

- More load: scale the API horizontally (it is stateless), run Postgres separately (a managed EU provider).
- Enterprise requirements: move the agent logic to Bedrock AgentCore / Azure Foundry (blueprint C/D);
  the tools remain usable as MCP servers.
- Strict data protection: self-host the models (vLLM) behind the same provider abstraction (blueprint E).
