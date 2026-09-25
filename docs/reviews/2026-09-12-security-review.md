# Security and architecture review — 2026-09-12

This review audits the `ai-app-starter` template — code, tests, CI, Docker Compose, and the checked-in docs (`README.md`, `CLAUDE.md`) plus the
ten draft ADRs in `docs/adr/` — as of commit `78d117f`. It was produced by 169 agents: eight independent lens finders (tenant-isolation,
authn-authz, agent-surface, secrets-deploy, data-protection, docs-vs-code, tests-ci, library-facts) each read the repo and the installed library
source directly, a completeness critic named further gaps the lenses missed, and every resulting finding then went through adversarial
verification against three separate tests — correctness of the citation, real-world exploitability, and whether the assigned severity and fix are
calibrated — before being counted as confirmed.

Of 145 raw findings evaluated (121 confirmed, 24 rejected), 121 survived verification: 19 High, 52 Medium, 50 Low as originally filed, several of
which are the same underlying fact reported by more than one lens and are merged below. Of the 24 rejected findings, 20 had their underlying
facts confirmed correct by the verifier and were downgraded only on severity, exploitability, or because the ADRs already resolve the question
raised; 4 were refuted because a citation or the central technical claim was actually wrong. A further 104 checks came back clean ("verified as
fine"). Three headline conclusions: (1) the RLS/roles/grants core is sound and well-tested, but the deployment wrapped around it — shared `.env`
files, fail-open auth defaults, no CI security gates — undermines it; (2) every safety net the agent surface needs (writing-tool confirmation,
usage limits, an audit trail, trustworthy conversation history) is either dead code or wholly absent, and the draft ADRs already know it; (3)
several compliance and architecture claims in the checked-in README and CLAUDE.md (GDPR/EU-residency, per-user permissions, "never holds the
superuser DSN") are already false against the code today, independent of whether the draft ADRs are ever accepted.

## How to read this

- **kind** — `bug`: the code contradicts its own stated behaviour or documentation; `gap`: a documented rule or a reasonable expectation has no
  implementation at all; `decision`: an open architectural fork the owner has not resolved (or the ADRs resolve on paper but the code hasn't
  caught up to yet).
- **severity** — `high`: a realistic path to cross-tenant data exposure, auth bypass, or a documented invariant actively violated in the shipped
  deployment; `medium`: a real gap with a plausible but conditional path to harm, or a significant claim that is false; `low`: a real defect with
  limited blast radius, a hardening nitpick, or a documentation nit.
- **Refuted items are not "wrong" by default.** The verifier is a single combined pass that checks correctness, exploitability, and
  severity/recommendation together, and refutes a finding if *any one* of those three checks fails — so most refuted items had their underlying
  facts confirmed and were downgraded only on severity, exploitability, or because an ADR already answers the question; only four were refuted
  because a citation or the core technical claim was itself incorrect. Section 4 marks each case explicitly.
- Every finding below cites `file:line` evidence as read against commit `78d117f` of this repo.

## Findings by severity

### High

- **API container holds the DB superuser DSN** (bug, tenant-isolation; found independently by 4 lenses) — docker-compose.yml and
  docs/deployment.md say the api container never holds the superuser DSN, but its `env_file: .env` injects `POSTGRES_PASSWORD` and
  `DATABASE_URL_MIGRATIONS` anyway, and `Settings` loads the migrations DSN into the api process; evidence: docker-compose.yml:24-25,36-38;
  docs/deployment.md:16; .env.example:10,15; app/config.py:21; fix: give the api service an explicit `environment:` allowlist instead of the
  whole `.env`, and stop `Settings` from holding `database_url_migrations` at all.
- **/api/chat's body-size cap is bypassed by chunked transfer encoding** (bug, agent-surface; found independently by 3 lenses, also confirmed by
  the completeness critic) — the 413 guard only checks `Content-Length`, which is absent on a chunked request, so `request.body()` reads an
  unbounded body into memory; evidence: app/api/chat.py:26-27; fix: enforce the cap on bytes actually read (stream `request.stream()` and abort
  past `MAX_BODY_BYTES`), or add a body-limit ASGI middleware.
- **Fail-open auth defaults ship as the actual defaults** (bug, authn-authz; found independently by 2 lenses) — `ENVIRONMENT=dev` and
  `AUTH_MODE=dev-headers` are both the code default and the `.env.example` value, and the startup guard only raises when environment is
  explicitly non-dev/test, so a copied `.env.example` deployed unmodified runs header-based tenant impersonation; evidence: app/config.py:17,31;
  app/main.py:25-31; .env.example:4; fix: make the defaults fail-closed (no default, or default `prod`) so a forgotten switch can't silently
  ship.
- **Langfuse captures full prompts and document content by default** (bug, agent-surface; found independently by 2 lenses) —
  `logfire.instrument_pydantic_ai()` is called with no arguments, so `include_content=True`/`include_binary_content=True` apply and full prompts,
  tool arguments, and document snippets flow into tracing; evidence: app/observability.py:37; pydantic_ai/models/instrumented.py:72-74; fix:
  default `include_content=False` and make content capture an explicit, tenant-scoped opt-in. Decided by: ADR-0008.
- **No `UsageLimits` on any agent run** (gap, agent-surface; found independently by 3 lenses) — no call site passes `UsageLimits`, so only
  pydantic-ai's default `request_limit=50` applies, with unbounded tokens and tool calls on top of a client-controlled 200 KB history; evidence:
  app/agents/assistant.py:44-49,67-83; app/api/chat.py:29-36; fix: pass an explicit `UsageLimits(request_limit=…, tool_calls_limit=…,
  total_tokens_limit=…)` from one shared helper. Decided by: ADR-0009.
- **The only RLS integration test cannot run on the owner's own checkout** (bug, tenant-isolation; found independently by 2 lenses) — pgserver's
  `psql()` shells out unquoted, and the checkout path contains a space, producing 4 ERRORS locally, not skips; evidence:
  tests/test_rls_integration.py:37-42; pgserver/postgres_server.py:247; fix: bootstrap the role via a list-form subprocess call or a direct
  asyncpg/SQLAlchemy connection instead of `server.psql()`.
- **Default privileges expose every future table before its RLS exists** (gap, tenant-isolation) — `01-init.sh` grants full DML to `app` on every
  table in `public` by default, so a new table is fully readable/writable the moment it's created, before any migration adds RLS, with no
  schema-invariant test to catch a forgotten policy; evidence: docker/postgres/01-init.sh:9; migrations/versions/0001_initial.py:103-104; fix:
  drop the blanket `ALTER DEFAULT PRIVILEGES` and grant explicitly per table, with a test asserting no non-allowlisted table is app-readable
  without RLS.
- **RLS's threat model is undocumented** (decision, tenant-isolation) — the `app` role can freely retarget `app.tenant_id` via `set_config`, so
  RLS here defends only against a developer mistake, not a compromised or SQL-injected app process, and nothing states this as an accepted
  residual risk; evidence: app/db/session.py:47-53; README.md:23; fix: write the residual risk explicitly into the architecture ADR's
  Consequences section. Decided by: ADR-0002.
- **/api/chat streams internal exception text to the client** (bug, agent-surface) — an unhandled exception mid-run reaches pydantic-ai's Vercel
  event stream, which yields `str(error)` with no re-raise or server-side log, so provider error bodies, SQL, and config hints can reach the
  browser; evidence: app/api/chat.py:29-36; pydantic_ai/ui/_event_stream.py:267,329; fix: wrap the run with an `on_error` hook that logs the real
  exception and emits a generic message keyed by `request_id`.
- **The LiteLLM gateway container receives every secret in the stack** (gap, secrets-deploy) — the third-party, floating-tag `litellm` image gets
  `env_file: .env` wholesale (DB superuser password, app DSN, Langfuse secret, all provider keys) and shares the default network with postgres,
  reaching it directly; evidence: docker-compose.yml:51-59; fix: give litellm an explicit environment allowlist, put it on its own network
  segment away from postgres, and pin the image to a release tag plus digest. Decided by: ADR-0009.
- **Per-tenant LiteLLM virtual keys are documented but don't exist in code** (bug, data-protection; found independently by 2 lenses) — deployment
  docs and `.env.example` promise minting per-tenant virtual keys with budgets, but `app/llm.py` uses one fixed process-wide `litellm_api_key`
  with no way to vary it per request; evidence: docs/deployment.md:30-34; app/config.py:24-25; app/llm.py:24-26; fix: implement per-tenant key
  resolution from a tenant repository, or drop the claim until it exists. Decided by: ADR-0009.
- **No schema-wide test enforces the "every table has RLS" rule** (gap, tests-ci) — the rule that every new table gets FORCE RLS and a policy is
  enforced only by a comment checklist in the migration template; nothing queries `pg_class`/`pg_policies`; evidence:
  migrations/script.py.mako:7-8; fix: add a test iterating every public table asserting `relrowsecurity`/`relforcerowsecurity` plus a policy,
  against an explicit allowlist.
- **No startup check verifies `DATABASE_URL` connects as a non-superuser role** (gap, critic-gap) — the FastAPI lifespan never queries the
  connected role's `rolsuper`/`rolbypassrls`, so a misconfigured `DATABASE_URL` pointing at the superuser would silently bypass RLS for every
  request; evidence: app/main.py:35-40; app/db/session.py:28-33; fix: after engine creation, query `pg_roles` for the current role and raise at
  startup if superuser or bypassrls is true, mirroring `check_auth_mode`.
- **Control-plane facts on `tenants` have no protected home** (decision, critic-gap) — the `app` role has table-level `UPDATE` on `tenants`,
  covering every current and future column, so a compromised app process could rewrite its own `isolation_tier` or `residency` once those columns
  exist; evidence: migrations/versions/0001_initial.py:96-109; CONTEXT.md:55-58; fix: split control-plane columns into a table or privilege set
  `app` can only SELECT, writable only by the owner/provisioning role.
- **A per-tenant secret store is assumed by four ADRs but never designed** (decision, critic-gap) — ADR-0002, 0005, 0008, and 0009 all reference
  "the secret store keyed by tenant" as if it already exists, but none specifies what it is or which role may decrypt which tenant's entry;
  evidence: docs/adr/0002:36-37; 0009:34-37; 0005:33-37; fix: decide the actual mechanism (external secret manager vs. an encrypted column)
  before implementing any of the four ADRs that depend on it.
- **The "owner role" for migrations is literally the Postgres superuser** (bug, critic-gap) — `01-init.sh` creates only the `app` role;
  `DATABASE_URL_MIGRATIONS` points at `POSTGRES_USER: postgres`, the cluster's real bootstrap superuser, contradicting the migration's own "owner
  role, not superuser" framing; evidence: docker/postgres/01-init.sh:1-11; docker-compose.yml:5-7,30; migrations/versions/0001_initial.py:7-9;
  fix: create a real non-superuser, NOBYPASSRLS owner/provisioner role and point migrations/seed at it instead.

### Medium

- **Docs claim per-user permissions; isolation is tenant-wide only** (gap, authn-authz; found independently by 2 lenses) — README and the tool
  docstring say tools run with "the current user's permissions", but no table has a per-user column and `app.user_id` is set and never read by
  any policy or query; evidence: README.md:23; app/agents/assistant.py:56; app/repositories/documents.py:49-63; fix: correct the wording to "the
  tenant's documents", or add a real per-user visibility column and policy. Decided by: ADR-0004.
- **Per-tenant model override is documented but dead code** (bug, agent-surface; found independently by 3 lenses) — CLAUDE.md and `app/llm.py`'s
  docstring say `tenants.settings["model"]` overrides the default, but no caller ever constructs `AssistantDeps` with a `model_name` read from
  tenant settings; evidence: app/llm.py:6; app/api/agents.py:39,46; app/api/chat.py:28; fix: implement the lookup against an allowlist of gateway
  aliases, or remove the claim. Decided by: ADR-0009.
- **No rate limiting on the agent endpoints** (gap, authn-authz; found independently by 2 lenses) — the only registered middleware is CORS;
  nothing throttles `/agents/assistant/*` or `/api/chat` per tenant or user, despite docs/mobile.md listing "per-user rate limits" as a backend
  duty; evidence: app/main.py:43-58; app/api/agents.py:37-51; app/api/chat.py:24-36; fix: add a per-(tenant_id, user_id) limiter dependency on
  all three routes. Decided by: ADR-0009.
- **No seam for routing a tenant to its own database** (decision, tenant-isolation; found independently by 4 lenses) — `app/db/session.py` uses
  one module-global engine from a single `DATABASE_URL`; `tenants.settings` (UPDATE-able by `app`) is the only per-tenant config slot and must
  never hold a DSN; evidence: app/db/session.py:24-33; app/config.py:20; fix: keep shared-DB+RLS as the default but let `tenant_session(ctx)`
  resolve its engine from a control-plane registry, not from `tenants.settings`. Decided by: ADR-0002.
- **MCP server has one process-wide identity and no auth** (gap, authn-authz; found independently by 4 lenses) — `context_provider` is a
  zero-argument callable reading two env vars with no check of `ENVIRONMENT`, so switching transport off stdio would expose the seeded tenant's
  documents to any network client, and no seam carries a per-connection identity; evidence: app/mcp/server.py:33-37,43,49; fix: refuse the
  env-based identity outside dev/test (mirroring `check_auth_mode`) and change the seam to take a per-connection MCP access token. Decided by:
  ADR-0005.
- **No audit trail of agent activity per tenant** (gap, agent-surface; found independently by 2 lenses) — nothing persists who asked what, which
  documents were retrieved, or what the agent answered; the only record is optional Langfuse tracing, silently disabled when `logfire` isn't
  installed; evidence: app/api/chat.py:29-36; app/agents/assistant.py:66-83; fix: add an `agent_runs`/`tool_calls` table under RLS, written from
  a shared post-run hook. Decided by: ADR-0005.
- **CI has no supply-chain or security gates** (gap, secrets-deploy; found independently by 2 lenses) — the workflow only runs `uv sync`, ruff,
  and pytest: no dependency-vulnerability audit (pip-audit already flags 3 vulnerabilities in a locked transitive dependency), no secret
  scanning, no Docker build/image scan, no lockfile enforcement, and no top-level `permissions:` block; evidence: .github/workflows/ci.yml:1-22;
  fix: add pip-audit, gitleaks, a Docker build+trivy scan, `uv lock --check`, a `permissions:` block, and SHA-pin actions.
- **All tenants share one Langfuse project; tenant_id isn't a real span attribute** (gap, data-protection; found independently by 2 lenses) — one
  Langfuse key pair means one shared project across every tenant, and `tenant_id` is only inside a JSON `metadata` string on the root run span,
  not a flat attribute other spans carry, which also blocks per-tenant trace deletion; evidence: app/observability.py:20-30;
  pydantic_ai/capabilities/instrumentation.py:260-261; fix: propagate tenant_id/user_id as OTel baggage so every span carries them as flat
  attributes. Decided by: ADR-0010.
- **The observability extra is never installed, so tracing is silently off** (bug, secrets-deploy; found independently by 4 lenses) — the Docker
  image builds with `uv sync --no-dev` (no `--extra observability`), and even the local venv has only the `logfire_api` no-op shim, so
  `setup_observability` warns and returns `False` instead of tracing, contradicting rule 7 ("every agent run is traced"); evidence:
  Dockerfile:6,8; pyproject.toml:21; app/observability.py:31-35; fix: install the extra in the image, and raise (not warn) when Langfuse is
  configured but the package is missing. Decided by: ADR-0008.
- **Client-supplied conversation history is trusted** (decision, agent-surface; found independently by 2 lenses) — `/api/chat` passes no
  `message_history`, so the whole conversation — including earlier assistant turns and tool results — comes from the request body; a forged
  `search_documents` result from the browser reaches the model verbatim; evidence: app/api/chat.py:29-36;
  pydantic_ai/ui/vercel_ai/_adapter.py:384-590; fix: persist conversations server-side under RLS and use only the last user message from the
  client body. Decided by: ADR-0006.
- **No provisioning path for the `app` role on managed Postgres** (gap, tenant-isolation) — the migration wraps every GRANT/REVOKE in `IF EXISTS
  (SELECT 1 FROM pg_roles WHERE rolname = 'app')` and silently skips them all if the role is missing, with no documented creation step for
  managed Postgres; evidence: migrations/versions/0001_initial.py:92,98-107; fix: make the grants unconditional and ship a documented
  role-creation script for managed Postgres.
- **RLS integration test misses several core cases** (gap, tenant-isolation) — the four existing tests never touch `users`/`tenants` policies,
  cross-tenant UPDATE/DELETE, a warm pooled connection with no context, or `DocumentRepository.add`; evidence:
  tests/test_rls_integration.py:97-158; fix: add the five missing cases using the existing fixtures (each verified to pass today).
- **X-Tenant-Id/X-User-Id are never validated against any table** (gap, authn-authz) — `get_context` builds `RequestContext` straight from parsed
  UUIDs with no membership check; any random UUID pair is accepted; evidence: app/deps.py:35-40; tests/test_api.py:36-43; fix: resolve the
  authenticated principal to a real membership row before building the context. Decided by: ADR-0003.
- **No model for machine/agent identities** (decision, authn-authz) — `RequestContext` requires a human `user_id`, and the MCP server has no
  principal model at all, so there's no way for a scheduled job or a customer's own automation to authenticate; evidence: app/context.py:16-17;
  app/mcp/server.py:33-37; fix: add an `api_keys`/agent-identity table with tenant-scoped, hashed, revocable credentials. Decided by: ADR-0005.
- **Tool results enter the prompt unframed as authoritative data** (gap, agent-surface) — the agent's instructions never state that
  `search_documents` output is untrusted data, and the tool result is handed to the model verbatim, a prompt-injection surface per CLAUDE.md rule
  4; evidence: app/agents/assistant.py:38-42,52-63; fix: add an explicit instruction that search results are untrusted data, and consider a
  clearly delimited envelope.
- **The confirmation flow for writing tools doesn't exist** (decision, agent-surface) — CLAUDE.md rule 4 requires a confirmation step before any
  writing tool, but nothing in `app/` uses pydantic-ai's `requires_approval`/`DeferredToolRequests` mechanism, and the one-shot
  `/agents/assistant/*` endpoints have no way to carry an approval round-trip; evidence: CLAUDE.md:44-48; fix: mark writing tools
  `requires_approval=True`, keep approvals on `/api/chat`, and back them with a server-side pending-action record. Decided by: ADR-0007.
- **`get_model()` accepts any string with no validation** (decision, agent-surface) — a tenant-supplied model name would resolve to `TestModel`,
  any provider the server holds credentials for, or any LiteLLM alias, with no allowlist gate; evidence: app/llm.py:18-28;
  pydantic_ai/models/__init__.py:1488ff; fix: validate `model_name` against an explicit allowlist before calling `infer_model`. Decided by:
  ADR-0009.
- **LiteLLM's virtual-key setup can't work as shipped** (bug, secrets-deploy) — the litellm container inherits the app's asyncpg `DATABASE_URL`,
  which LiteLLM interprets as its own Prisma database rather than a dedicated one, so per-tenant virtual keys have nowhere real to live;
  evidence: docker-compose.yml:54; .env.example:13; docker/litellm/config.yaml:22; fix: provision a dedicated Postgres database/role for LiteLLM.
  Decided by: ADR-0009.
- **Default DB passwords ship baked into every config surface** (gap, secrets-deploy) — `postgres`/`app` are literal defaults in compose,
  `.env.example`, and `Settings`, with no production guard, and a password containing a quote breaks the unescaped SQL in `01-init.sh`; evidence:
  docker-compose.yml:6,8; app/config.py:20-21; docker/postgres/01-init.sh:5-6; fix: require passwords with no default outside dev; pass the
  password via `psql -v` instead of string interpolation.
- **Every base image is a floating tag** (gap, secrets-deploy) — `python:3.12-slim`, `uv:latest`, `pgvector:pg17`, and `litellm:main-stable` are
  all mutable tags with no digest pin, so the gateway holding every provider key can auto-update unreviewed; evidence: Dockerfile:1-2;
  docker-compose.yml:3,52; fix: pin every image to a version tag plus digest and add Renovate/Dependabot for docker + uv.
- **Decision: how should secrets be delivered to containers?** (decision, secrets-deploy) — `env_file: .env` hands every service the entire file;
  pydantic-settings already supports `secrets_dir` with no code changes; evidence: docker-compose.yml:28,38,54; app/config.py:14; fix:
  per-service `environment:` allowlists now, and file-backed Docker `secrets:` + `secrets_dir="/run/secrets"` for production. (Open decision — no
  ADR covers it; see below.)
- **Default LLM route sends inference to US Anthropic, not the EU alias** (decision, data-protection) — `llm_model` defaults to the direct
  `anthropic:claude-sonnet-4-5` endpoint; the `claude-eu` (Bedrock eu-central-1) alias exists in the gateway config but is never the documented
  default; evidence: app/config.py:23; docker/litellm/config.yaml:4-11; fix: make `claude-eu` the documented production default. Decided by:
  ADR-0008.
- **No per-tenant usage accounting or cost isolation** (gap, data-protection) — `result.usage()` is discarded after every run, no `usage_limits`
  are passed, and no rate limiting exists, so cost isolation between tenants does not exist; evidence: app/api/agents.py:38-40;
  app/agents/assistant.py:67-73; fix: persist usage per run into an RLS-protected table, pass `UsageLimits`, and add a per-tenant rate limit.
  Decided by: ADR-0009.
- **No data-retention or tenant-erasure path** (gap, data-protection) — there is no delete anywhere in `app/`; `scripts/` only ever creates a
  tenant; traces holding content have no documented retention; evidence: app/repositories/documents.py; migrations/versions/0001_initial.py:105;
  fix: ship a `delete_tenant.py` script using the existing `ON DELETE CASCADE`, and document Langfuse retention. Decided by: ADR-0010.
- **`check_auth_mode`'s wiring in lifespan is never exercised by tests** (gap, tests-ci) — the API test client uses `httpx.ASGITransport`, which
  never sends ASGI lifespan events, so the only place the guard actually runs in production is untested; evidence: app/main.py:35-40;
  tests/test_api.py:18; fix: drive `app.router.lifespan_context(app)` directly in a test with prod-like settings.
- **/api/chat and /agents/assistant/stream have zero HTTP tests** (gap, tests-ci) — no test exercises the Vercel adapter path, SSE output, or the
  400/501/CORS branches; a pydantic-ai signature change would surface only at runtime; evidence: tests/test_api.py:22-44; fix: add auth,
  happy-path, and error-branch tests for both endpoints.
- **`PermissionError` from `require_role` becomes an unhandled 500** (gap, tests-ci) — no exception handler is registered for it, so the moment a
  role check is added anywhere, a failed check crashes with 500 instead of a defined 403; evidence: app/context.py:24-26; app/main.py:43-58; fix:
  register a `PermissionError` → 403 exception handler and test it.
- **Cross-tenant UPDATE/DELETE and tenants/users visibility are untested** (gap, tests-ci) — these all currently hold correctly (verified
  empirically) but have no regression test, so a future change could silently break them; evidence: tests/test_rls_integration.py:97-158; fix:
  add the five prototype tests described in the finding.
- **No DB-side audit logging for the privileged migrate/seed role** (gap, critic-gap) — nothing enables `pgaudit` or `log_statement` for the
  connection running with owner/superuser privileges, so what it actually executes is invisible; evidence: grep of docker/, migrations/, app/ for
  pgaudit|log_statement is empty; fix: enable `log_statement` (or pgaudit) scoped to the owner role in `01-init.sh`.
- **ADR-0010's future CLI still plans to reuse the superuser DSN, with no role or audit trail of its own** (decision, critic-gap) — no ADR
  specifies what "the tool needs the highest credentials in the system and therefore its own hardening" (ADR-0010's own words) actually means;
  evidence: docs/adr/0010-tenant-lifecycle.md:18-19,49; fix: run the future CLI as the scoped owner role recommended above, with its own
  structured log. (Open decision — see below.)
- **Tenant-selection mechanism for a request is undecided** (decision, critic-gap) — no ADR states whether the tenant comes from a header, path,
  subdomain, or a verified token claim; evidence: app/deps.py:24,30-33,36; docs/adr/0003:22-23; fix: put the tenant in the URL path and bind the
  access token's audience to it, checked alongside the membership row. (Open decision — see below.)
- **README's GDPR claim is already false against the checked-in code** (bug, critic-gap) — independent of the draft ADRs, embeddings always leave
  for OpenAI's US endpoint and Langfuse exports full content by default today, contradicting "GDPR is covered via EU regions + DPA"; evidence:
  README.md:122; app/embeddings.py:20-22; fix: correct or scope the README claim now, independently of ADR-0008's status. Decided by: ADR-0008.
- **No behavioural/eval tests for agent instructions or prompt-injection resistance** (gap, critic-gap) — the only agent tests assert a tool was
  called, never that the model follows its instructions or resists an adversarial tool result; evidence: tests/conftest.py:39-43; fix: add a
  small eval suite using `pydantic-evals` or `FunctionModel` adversarial cases.
- **Bumping LLM_MODEL has zero regression protection** (gap, critic-gap) — README instructs derived projects to bump the model on every fork, but
  every agent test runs against `TestModel`, which never calls a real model; evidence: README.md:15-17; fix: add a small real-model eval harness,
  gated behind a secret/flag.
- **Document ingestion has no code path or provenance schema** (gap, critic-gap) — only `search` is implemented; `DocumentRepository.add` is
  never called outside a test, and the README quickstart implies documents already exist; evidence: app/repositories/documents.py:28-47;
  scripts/seed.py:22-44; fix: either scope ingestion explicitly out of the starter, or ship a minimal ingestion tool through the writing-tool
  approval flow.
- **No Content-Type enforcement is a latent CSRF vector for derived projects using cookie auth** (gap, critic-gap) — the Vercel adapter never
  checks `Content-Type` before parsing the body, which combined with docs/frontend.md's cookie-auth advice and credentialed CORS creates a CSRF
  path for a project that follows the docs; evidence: pydantic_ai/ui/_adapter.py:329-331; fix: add a `Content-Type: application/json` check as
  shared middleware before implementing cookie auth.
- **No wall-clock deadline on model calls** (gap, critic-gap) — no `model_settings.timeout` is ever passed, so a stalled provider can hold a run
  (and a streaming client) open for up to the SDK's ~30-minute default; evidence: app/agents/assistant.py:66-83; fix: pass an explicit short
  `ModelSettings(timeout=…)` sized under the reverse-proxy's own timeout.

### Low

- **OTLP endpoint is set via `os.environ.setdefault`, so a pre-set platform var silently wins** (bug, data-protection; found independently by 2
  lenses) — a platform-level `OTEL_EXPORTER_OTLP_ENDPOINT` would silently redirect content-bearing traces and could attach the Langfuse
  Basic-auth header to a foreign endpoint; evidence: app/observability.py:27-30; fix: construct the OTLP exporter explicitly instead of via
  environment variables.
- **RLS NULL-vs-empty-string docstring claim only holds on a fresh connection** (bug, tenant-isolation) — on a reused pooled connection
  `current_setting` returns `''`, not `NULL`, so every policy evaluation errors instead of blocking cleanly; evidence: app/db/session.py:5-6;
  fix: write policies as `NULLIF(current_setting(...), '')::uuid` and correct the docstring.
- **No per-role resource limits for `app`** (gap, tenant-isolation) — no `statement_timeout`, `idle_in_transaction_session_timeout`, or
  connection limit is set, so one tenant's heavy query is a noisy neighbor for all others; evidence: docker/postgres/01-init.sh:6-10; fix: `ALTER
  ROLE app SET statement_timeout=...` and a connection limit.
- **Two disconnected role models never agree** (bug, authn-authz) — `users.role` in the DB and `RequestContext.roles` from the `X-Roles` header
  are unrelated, and neither is enforced anywhere; evidence: migrations/versions/0001_initial.py:64; app/deps.py:39; fix: pick one source of
  truth for roles and remove or dev-only-mark `X-Roles`. Decided by: ADR-0004.
- **No user/tenant lifecycle state** (gap, authn-authz) — there is no `disabled_at`/`suspended_at` column and no revocation model anywhere;
  evidence: migrations/versions/0001_initial.py:41-46, 60-67; fix: add lifecycle columns and reject in context resolution when set. Decided by:
  ADR-0010.
- **No audit trail independent of optional tracing** (gap, authn-authz) — `request_id` is generated per request but never returned to the client
  or logged anywhere; evidence: app/context.py:19; fix: echo `X-Request-Id` and log a structured line per authenticated request.
- **MCP tool errors return raw exception text** (gap, agent-surface) — no try/except wraps the MCP tool call, so SQL with parameters or embedding
  API error bodies reach the MCP client; evidence: app/mcp/server.py:46-50; fix: catch and return a generic `is_error` message, matching the API
  error path.
- **/agents/assistant/stream has no error path** (bug, agent-surface) — a mid-run exception truncates the SSE stream with HTTP 200 and no
  terminal event; evidence: app/api/agents.py:45-49; fix: wrap the generator and emit `event: error` before closing.
- **/agents/assistant/run keeps running after client disconnect** (gap, agent-surface) — only streaming responses are cancelled on disconnect;
  the plain `/run` handler has no equivalent; evidence: app/api/agents.py:37-40; fix: bound `/run` with a timeout or prefer streaming.
- **`repr(Settings())` prints every secret in clear text** (gap, secrets-deploy) — API keys, the Langfuse secret, and the superuser DSN are plain
  `str` fields; evidence: app/config.py:21,25, 28-29,35; fix: use `pydantic.SecretStr` for every credential field.
- **Container runtime hardening gaps** (gap, secrets-deploy) — `/app` (incl. `.venv`) is writable by the runtime user, no `HEALTHCHECK`, no
  restart policy, no resource limits, no read-only rootfs/cap_drop; evidence: Dockerfile:8; docker-compose.yml:36-48; fix: chown to root, add
  HEALTHCHECK, `restart: unless-stopped`, resource limits, and rootfs hardening.
- **Uncommitted exec-bit change on `01-init.sh`** (decision, secrets-deploy) — the working tree shows mode 100644 vs. the committed 100755; the
  postgres entrypoint sources non-executable scripts identically either way; evidence: `git diff` mode 100755→100644; fix: commit the 100644 mode
  deliberately with a comment that it's sourced, not executed.
- **`.gitignore`/`.dockerignore` exclude only `.env`, not `.env.*`** (gap, secrets-deploy) — `.env.prod`/`.env.local` variants would not be
  excluded; evidence: .gitignore:4; .dockerignore:2; fix: exclude `.env*` and explicitly re-allow `.env.example`.
- **`LITELLM_MASTER_KEY` is empty by default with no guard** (gap, secrets-deploy) — `.env.example` ships it empty, and compose has no `:?` guard
  requiring it to be set; evidence: .env.example:27; docker-compose.yml:51-59; fix: use `${LITELLM_MASTER_KEY:?required, must start with sk-}`.
- **No TLS guidance for the DB connection** (gap, secrets-deploy) — asyncpg defaults to `sslmode=prefer` (opportunistic, unverified), and no DSN
  in the repo carries `ssl=require`; evidence: .env.example:13,15; asyncpg/connect_utils.py:656; fix: document `?ssl=require` for any non-local
  `DATABASE_URL`.
- **No encryption-at-rest / per-tenant key story** (decision, data-protection) — nothing is configured; pgvector's fixed-dimension column makes
  per-tenant encryption effectively a DB-per-tenant decision; evidence: docker-compose.yml:9-11; migrations/versions/0001_initial.py: 74-83; fix:
  rely on provider disk encryption and document it; treat BYOK as the DB-per-tenant trigger. Decided by: ADR-0002.
- **DEBUG logging of the OpenAI/Anthropic SDKs would leak request bodies** (gap, data-protection) — nothing pins those loggers away from DEBUG,
  and the SDK logs full request bodies at that level; evidence: openai/_base_client.py:523-535; fix: pin `openai`/`anthropic`/`httpx` loggers to
  INFO in `create_app()`.
- **"Cache keys include the tenant_id" describes a cache that doesn't exist** (gap, docs-vs-code) — README and CLAUDE.md state this as a property
  of the code; the only `@lru_cache` uses are unrelated to tenancy; evidence: README.md:75-76; app/config.py:45; fix: rephrase as a rule for any
  future cache.
- **RequestContext docstring mentions jobs and per-MCP-connection creation that don't exist** (gap, docs-vs-code) — no job runner exists, and the
  MCP context is per-tool-call from a process-wide env identity, not per connection; evidence: app/context.py:3-4; fix: trim the docstring to
  what actually exists.
- **Test conventions in CLAUDE.md aren't met by the template's own tests** (bug, docs-vs-code) — `DocumentRepository.add` has no test, and
  `FunctionModel` is mentioned but never used; evidence: CLAUDE.md:33-34; fix: add the missing repository test and either use `FunctionModel` or
  drop the mention.
- **chat.py's body-cap comment says "not unbounded" but chunked bodies bypass it** (bug, docs-vs-code) — the same underlying bug as the merged
  High chunked-body finding; evidence: app/api/chat.py:19-27; fix: fix the check (see the High finding above) and correct the comment.
- **`.mcp.json.example` relies on the process cwd for `.env`** (gap, docs-vs-code) — this works for Claude Code but isn't guaranteed for Claude
  Desktop as advertised; evidence: .mcp.json.example:110-115; app/config.py:14; fix: add a `cwd`/`--directory` example, or fail fast when
  required settings are missing.
- **The `limit` clamp test doesn't test the clamp** (gap, tests-ci) — `test_assistant`'s assertion exercises TestModel's default argument, not
  `app/tools/documents.py`'s actual clamp; evidence: tests/test_assistant.py:12; fix: add a direct unit test of the clamp function.
- **Tests aren't hermetic against the developer's own `.env`** (gap, tests-ci) — a local `AUTH_MODE=jwt` makes the API test suite fail; evidence:
  app/config.py:14,45-47; fix: add an autouse fixture that pins test environment variables and clears the settings cache.
- **`app` has full DML on `alembic_version`** (gap, tests-ci) — the same default-privileges leak (see the High "default privileges" finding) also
  covers Alembic's own bookkeeping table; evidence: docker/postgres/01-init.sh:9; fix: `REVOKE ALL ON alembic_version FROM app` in a migration.
- **MCP server has no tests at all** (gap, tests-ci) — the context-from-env guard, tool registration, and clamp path are all untested; evidence:
  grep of tests/ for `mcp` is empty; fix: add `tests/test_mcp.py` covering the guard and the registered tool.
- **chat.py uses a deprecated starlette status constant on an untested path** (bug, tests-ci) — `HTTP_413_REQUEST_ENTITY_TOO_LARGE` is deprecated
  in starlette 1.6; the 413 branch has no test; evidence: app/api/chat.py:27; fix: use `HTTP_413_CONTENT_TOO_LARGE` and add the test.
- **The client-chosen Vercel chat `id` becomes the trace `conversation_id` with no tenant scoping** (gap, library-facts) — `dispatch_request`
  never passes `conversation_id`, so the library falls back to the client-supplied `id` verbatim; evidence: pydantic_ai/ui/vercel_ai/_adapter.py:
  302-305; fix: pass `conversation_id=f"{tenant_id}:{client_id}"`.
- **`CORS_ORIGINS` is one process-wide list, not scoped per tenant** (gap, critic-gap) — a single flat setting applies to every request
  regardless of tenant; evidence: app/config.py:18,41-42; fix: document the limitation, or make it per-tenant once tenant naming is decided.
- **Nine untracked "proposed" ADRs contradict the checked-in docs with no record of which decision is implemented** (gap, critic-gap) — `git
  status` shows CONTEXT.md and ADR-0002 through 0010 as untracked, each dated today, while CLAUDE.md/README are not updated to match; evidence:
  `git status`; docs/adr/0002-0010; fix: run an explicit accept/reject/defer pass over the ADRs and update CLAUDE.md/README accordingly.
- **README's quickstart search example can't work** (bug, critic-gap) — the seed script inserts a tenant and a user but zero documents, yet the
  next quickstart step asks the assistant about document content; evidence: README.md:24-28,39-43; scripts/seed.py:22-44; fix: add a
  document-seeding option, or change the example prompt.
- **Embedding calls carry no explicit timeout** (gap, critic-gap) — `embed()` inherits the OpenAI SDK's 600-second default plus 2 retries,
  blocking document search for up to ~30 minutes on a slow endpoint; evidence: app/embeddings.py:25-30; openai/_constants.py:7-8; fix: pass a
  short explicit timeout.
- **No egress allow-list/SSRF-guard seam exists** (gap, critic-gap) — no URL-fetching tool exists yet, and nothing in the architecture rules
  calls for a guard before the first one is added; evidence: grep of app/ for httpx/requests/ssrf is empty; fix: add a host allow-list helper
  before any URL-fetching tool ships.
- **docker-compose gives the api container unrestricted outbound access, with no network segmentation** (decision, critic-gap) — postgres, api,
  and litellm all share one default network; evidence: docker-compose.yml:1-65; fix: document an egress proxy/allow-list for production; keep one
  network for the starter's own local/dev use. (Open decision — see below.)
- **Swagger UI, ReDoc, and `/openapi.json` are public in every environment** (gap, critic-gap) — none of the three routes go through
  `get_context`; they're registered directly on FastAPI with its defaults; evidence: app/main.py:43-45; fix: gate them behind
  `settings.environment` unless deliberately kept public (related open decision below).
- **FORWARDED_ALLOW_IPS is unconfigured for the documented reverse-proxy deployment** (gap, critic-gap) — no value appears anywhere in
  `.env.example`/compose/docs, so uvicorn's default (127.0.0.1) may distrust the actual proxy hop from the docker bridge; evidence:
  docker-compose.yml:44-48; fix: add `FORWARDED_ALLOW_IPS` to `.env.example` with a note about the docker bridge subnet.
- **No tenant data-export/portability path** (gap, critic-gap) — the only lifecycle code is `scripts/seed.py`, which only creates; ADR-0010's
  planned erasure tool only removes data, never exports it first; evidence: docs/adr/0010:11,29-35; fix: add an `export` operation alongside
  ADR-0010's planned `create`/`suspend`/`delete`. Decided by: ADR-0010.
- **No sub-processor register despite the unqualified GDPR/DPA claim** (gap, critic-gap) — tenant content actually flows to Anthropic, OpenAI,
  AWS Bedrock, and Langfuse's host, none disclosed; evidence: README.md:122; app/config.py:23; app/embeddings.py:22; fix: generate
  `docs/subprocessors.md` from the same allow-list the residency guard will validate against. Decided by: ADR-0008.
- **Auth failures in `get_context` are never logged** (gap, critic-gap) — none of the 401/400/501 branches call `log.*`; evidence:
  app/deps.py:29-47; fix: add a `log.warning` in each rejection branch (without echoing raw header values).
- **RLS policy violations are invisible in production** (gap, critic-gap) — no exception handler recognizes the `DBAPIError` Postgres raises on a
  `WITH CHECK` failure; it would propagate as an unhandled 500 with no correlation to tenant/request; evidence:
  tests/test_rls_integration.py:117-131; fix: add a handler that logs RLS-shaped errors with tenant_id/request_id and counts them.
- **No SECURITY.md / vulnerability disclosure policy** (gap, critic-gap) — README invites PRs/ issues but nothing says how to privately report a
  vulnerability; evidence: README.md:133; fix: add a short SECURITY.md linked from the README.

## Refuted on severity, facts hold

The correctness of each of these was confirmed by the verifier; only the severity, the exploitability chain, or the "this is an open decision"
framing was rejected.

- **`app` can UPDATE/DELETE `alembic_version`** — the over-grant is real, but reaching it already requires holding the app role's own DB
  credentials, a worse compromise than the medium tenant-isolation narrative implied; downgraded to a low-severity hardening item (captured above
  as "app has full DML on alembic_version").
- **Connection pool never resets session-level GUCs** — the mechanism (SQLAlchemy's default reset-on-return is ROLLBACK, not `RESET ALL`) is
  real, but every shipped code path already sets `app.tenant_id` with `is_local=true` before querying, so nothing currently reachable is exposed;
  downgraded from medium to a purely theoretical, future-code-only low.
- **`01-init.sh` interpolates `APP_DB_PASSWORD` unescaped** — the quoting bug is real, but a password with a quote fails role creation closed
  (the container aborts on first boot); reframed as a script-hygiene bug, not a tenant-isolation security gap.
- **Decision: tenant_id source under `AUTH_MODE=jwt`** — the citations (a 501 stub) are accurate, but ADR-0003 already resolves this fork (global
  identity + per-request membership verification); downgraded from an open decision to an implementation-lag gap.
- **Decision: one person in several tenants** — the schema/RLS facts hold, but ADR-0003 already chose global identities + memberships, rejecting
  the finding's own recommended alternative.
- **Decision: per-customer identity providers** — the citations are accurate, but ADR-0003 already names this exact fork with an interim answer
  (one shared IdP) and a documented revisit trigger.
- **OpenAPI docs unauthenticated everywhere** — confirmed exactly, but docs/mobile.md documents `/openapi.json` as the intended public
  client-contract source, so today's exposure is closer to by-design than an oversight (captured above as its own Low finding regardless).
- **`check_auth_mode` only runs in lifespan** — confirmed literally, but the documented deployment command (`uvicorn app.main:app`) always runs
  lifespan by default, so the guard does fire in production; the gap is in the test harness, not the shipped deployment (captured above as its
  own Medium finding).
- **No platform-admin identity for tenant creation** — facts confirmed, but README already documents onboarding as arriving "with the second
  customer, not before"; no admin API exists to attack today.
- **`search_documents` is tenant-wide, not per-user** — facts confirmed, but ADR-0004 already decided tenant-wide visibility deliberately,
  rejecting per-document RLS for the template.
- **uvicorn ignores X-Forwarded-For behind the documented proxy** — the misconfiguration is real, but the claimed link to
  cost-abuse/rate-limiting is wrong: `RequestContext` is built from JWT claims, not client IP, so this doesn't block a tenant-keyed rate limiter
  (which is its own, separately reported, Medium gap above).
- **One compose file serves both dev and single-server production** — mostly holds (one overstatement: the DB port is loopback-only, not
  published), but docs/deployment.md already documents this as deliberate, and the one auth-critical setting is already hard-enforced by
  `check_auth_mode`.
- **Every query is embedded at OpenAI US regardless of route** — confirmed and real, but "high bug contradicting the EU-residency claim"
  overstates it: the GDPR claim pairs EU regions with a DPA covering SCC-based transfers, not physical-only processing; reframed as a
  decision/gap, not a high-severity bug (the broader, code-verified false claim is still reported above as its own Medium bug).
- **Langfuse host is free-form, could point outside the EU** — confirmed as written, but the docs never actually offer a non-EU option, so the
  "decision" isn't a real fork.
- **MCP hands snippets to whichever model the connecting client uses** — confirmed verbatim, but ADR-0005 already decides MCP is a production
  surface moving to OAuth/Streamable HTTP; the residual DPA point is inherent to the MCP protocol, not a template-specific fork.
- **RLS test errors instead of skipping on a space-containing checkout path** — confirmed and reproduced, but contingent on this developer's own
  checkout path and doesn't affect CI; the same underlying pgserver bug is already captured above as a confirmed High finding from two other
  lenses.
- **Fail-open defaults filed as an undecided "high" decision** — the same fact as the confirmed fail-open-defaults bug above, but here the
  exploitability chain requires the operator to skip a documented one-line deployment step, and the fail-open-for-dev pattern is already a
  documented, deliberate convention.
- **Confirmation flow "cannot work", filed as a fresh gap** — the library mechanics are confirmed, but ADR-0006 and ADR-0007 already document
  this exact behaviour and prescribe the fix in more depth than the finding itself.
- **logfire exports full content by default, filed as an open "decision"** — the same fact as the confirmed Langfuse-content finding above, but
  ADR-0008 already decided `include_content=False` as the target state; what remains is an implementation-lag gap, not an open decision.
- **`tenants.settings["model"]` called an unvalidated-tenant-input risk** — confirmed dead code, but the security narrative requires a feature
  that doesn't exist and isn't scheduled; downgraded to what the confirmed docs-mismatch finding above already says.

## Refuted as wrong

- **Vector index recommendation ignores RLS post-filtering** — the cited `CREATE INDEX` is commented-out example guidance, never executed; no
  approximate index exists today, so the claimed recall-loss mechanism doesn't apply to the shipped schema.
- **`require_role` raises `PermissionError`, surfacing as 500 not 403** — the FastAPI 500-on- unhandled-exception fact is right (and is captured
  above as its own confirmed Medium finding), but the specific pydantic-ai internals cited are wrong (wrong function; a precondition claimed
  always-false is actually always-true), and `require_role` has zero callers today.
- **`tenants.settings` mixes tenant and operator-owned facts in one untyped JSONB** — the underlying concern is real, but the write cites an
  ADR-0004 line that doesn't support it and infers a claim from CONTEXT.md the text never makes; no code today writes settings at all, making it
  a pre-implementation design fork rather than a present "bug" (the same underlying concern is captured more rigorously above as the confirmed
  "control-plane facts have no protected home" finding).
- **Process-wide Settings cache blocks secret rotation without a restart** — contains a false claim ("no `cache_clear()` call anywhere in the
  codebase" — the RLS test suite does call it), and restart-to-rotate is standard container behaviour, not a gap specific to this design;
  speculative about a not-yet-built feature.

## Verified as fine

**Tenant isolation (RLS core).** The `app` role's flags (`rolsuper=false`, `rolbypassrls=false`, `rolcreatedb=false`, no `CREATE` on `public`)
are verified on a live cluster; all three tables carry `ENABLE`+`FORCE` RLS with `FOR ALL` policies enforcing both `USING` and `WITH CHECK`
(confirmed even where only `USING` is written, since Postgres applies it as both for `FOR ALL`); cross-tenant UPDATE/DELETE affect 0 rows and a
`tenant_id` reassignment raises `InsufficientPrivilege`; `app` has no INSERT/DELETE on `tenants`, so the table's `ON DELETE CASCADE` is reachable
only by the owner role; `set_config(..., is_local=true)` is transaction-scoped, so a pooled connection never carries one request's tenant into
the next, and a mid-transaction `commit()` fails closed; `scripts/seed.py` and every repository write set `tenant_id` from context with no
f-string SQL anywhere in `app/`.

**Auth guard (dev-headers path).** `AUTH_MODE` is regex-validated at startup; the environment guard fails closed on any value other than exactly
`dev`/`test`; `AUTH_MODE=jwt` correctly 501s every route rather than serving a half-implemented check; missing/malformed headers get 401/400;
every agent and chat route depends on `Context` (only `/health` is unauthenticated, and it does nothing); the streaming endpoint resolves context
before any bytes are sent; CORS never combines a wildcard origin with credentials; `RequestContext` is an immutable frozen dataclass.

**Agent surface (library defaults).** The Vercel adapter already strips client-injected system prompts, disallowed file URLs, uploaded-file
references, and dangling trailing tool calls by default; the model cannot choose the tenant (it comes from `RunContext` deps, never tool
arguments); the `limit` clamp (1-20) is enforced in the shared tool function for both the agent and MCP paths; streaming endpoints do cancel on
client disconnect; `/agents/assistant/run`'s generic 500 doesn't leak provider/DB text (unlike `/api/chat`, reported above); tests run entirely
against `TestModel`/`FunctionModel`, never a real model call.

**Secrets and deployment.** Postgres, API, and LiteLLM ports are all bound to `127.0.0.1` only, matching the docs; `.env` is excluded from both
the image and git, and pydantic-settings silently tolerates a missing env file; the container runs as a non-root user; migrations run before the
API with `service_completed_successfully`, using the owner DSN only on the one-shot `migrate` service; `01-init.sh` correctly creates `app` as
`NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE`; no secret was found in git history; `uv.lock` is currently in sync with `pyproject.toml`, so
the lockfile-enforcement gap above is latent, not actively broken today.

**Data protection.** `logfire.configure(send_to_logfire=False)` sends nothing to Pydantic's own cloud; application code never logs
prompt/tool/document content itself; embeddings live under the same RLS policy as their source documents and cascade on tenant delete; tool
results are minimised before reaching the model (400-character snippets, id/title/score only); request payloads are bounded server-side for the
non-chunked case; the OpenAI client already honours `OPENAI_BASE_URL`, so an EU-hosted embedding endpoint is a configuration change away, not a
code change.

**Docs vs. code (everything not flagged in section 6).** The startup guard, loopback port bindings, owner-DSN usage in migrations/seed, the RLS
template's shape, the `app` role's NOSUPERUSER/NOBYPASSRLS flags, `tenant_session`'s `is_local` semantics, the shared
`app/tools/`-for-agent-and-MCP pattern, the "read-only tools only" claim, the MCP SDK import path, the Vercel adapter signature, the LiteLLM
provider construction, `.env.example`'s completeness against `Settings`, the version banner, and the CI/README description of the RLS test's skip
behaviour were all checked line-by-line against the installed library source and found accurate.

**Tests, CI, and library facts.** `ruff check`/`ruff format --check` are clean; `uv lock --check` passes; the full unit and RLS suites pass once
the pgserver quoting bug is worked around, with the exact catalog state (RLS flags, grants, cross-tenant semantics) matching what the code
claims; the `limit` clamp, `check_auth_mode`'s branch logic, and the CORS credentials guard are all correctly implemented and unit-tested; trace
metadata does carry identifiers only, exactly as documented; `pydantic_ai`'s own `request_limit=50` default does cap a runaway tool loop even
without an explicit `UsageLimits`, bounding the impact of the missing-limits finding above.

## Documentation claims that are false today

| Claim (file:line) | Reality (file:line) |
| --- | --- |
| "Keeps the superuser DSN out of the long-running api container" / "the api container never holds the superuser DSN" (docker-compose.yml:24-25; docs/deployment.md:16) | `api` service also has `env_file: .env`, injecting `POSTGRES_PASSWORD` and `DATABASE_URL_MIGRATIONS` (docker-compose.yml:36-38; app/config.py:21) |
| "tools that run with the logged-in user's permissions" (README.md:23) | Isolation is tenant-only; no user-scoped filter or column exists (app/repositories/documents.py:49-63) |
| "Cache keys include the `tenant_id`" (README.md:75-76; CLAUDE.md:53) | No cache exists in the codebase (app/config.py:45, the only `@lru_cache` use, is unrelated to tenancy) |
| RequestContext "Created once per request … per job, or per MCP connection" (app/context.py:3-4) | No job runner exists; MCP context is per-tool-call from a process-wide env identity, not per connection (app/mcp/server.py:43,49) |
| "For every repository function, a test with a second tenant" / "Agent tests with TestModel/FunctionModel" (CLAUDE.md:33-34) | `DocumentRepository.add` has no test; `FunctionModel` is never used anywhere (grep of tests/ and app/) |
| "not unbounded — mirrors the 20k prompt cap" (app/api/chat.py:19-21) | Only `Content-Length` is checked; a chunked-transfer body bypasses the cap entirely (app/api/chat.py:26-27) |
| ".mcp.json.example … Claude Code/Desktop" (CLAUDE.md:27) | Resolution of `.env` depends on the process's cwd (app/config.py:14), not guaranteed under Claude Desktop's working directory |
| "Then `uv sync --extra observability`" / "Every agent run is traced" (docs/deployment.md:23-26; CLAUDE.md rule 7) | The Docker image builds with `--no-dev` and never installs the extra; even the local venv has only the `logfire_api` no-op shim (Dockerfile:6,8) |
| "mint virtual keys with per-tenant budgets via the LiteLLM admin API" (docs/deployment.md:30-34; .env.example:26) | One fixed process-wide `litellm_api_key` is used for every tenant, and litellm gets the app's own asyncpg `DATABASE_URL` instead of a dedicated one (app/config.py:24-25; docker-compose.yml:54) |
| "GDPR is covered via EU regions + DPA" (README.md:122) | Every embedding call defaults to OpenAI's US endpoint, and Langfuse exports full prompt/document content by default (app/embeddings.py:20-22; app/observability.py:37) |
| Quickstart curl asks the assistant about document content (README.md:24-43) | `scripts/seed.py` inserts a tenant and a user but zero documents (scripts/seed.py:22-44) |

## Open decisions not yet covered by an ADR

> Written against ADR-0002 to ADR-0010. Later the same day the owner decided the second to sixth
> items below: ADR-0011 (control-plane schema, `app_owner` role, tenant secrets as files under
> `/run/secrets`, per-service environment allow-lists) and ADR-0012 (tenant in the URL path with
> token audience and membership agreeing). Wall-clock deadlines were folded into ADR-0009. The
> exec-bit, network-segmentation, and `/openapi.json` items are handled as fixes, not decisions.

- **Uncommitted exec-bit change on `docker/postgres/01-init.sh`.** Should the 100644 mode (script is sourced, not executed) be committed
  deliberately, or should the 100755 exec bit be restored? Finder's recommendation: commit 100644 deliberately with a comment explaining it's
  sourced — more robust than depending on the exec bit surviving sync/checkout.
- **Secrets delivery mechanism.** Should production secrets move to file-based Docker secrets read via pydantic's `secrets_dir`, with per-service
  `environment:` allowlists as a minimum now, or stay on `env_file`/environment variables indefinitely? Finder's recommendation: per-service
  allowlists immediately (closes the three env_file findings above cheaply), and Docker `secrets:` + `secrets_dir="/run/secrets"` for production.
- **Where do operator-owned control-plane facts live?** Should `isolation_tier`, `residency`, database alias, and model-allowlist selection on
  `tenants` be columns/tables the `app` role can only SELECT, with all writes going through the owner/migrations role, before
  ADR-0002/0008/0009's per-tenant fields are implemented? Finder's recommendation: yes — split the table or its privileges before implementing
  those ADRs, not after.
- **What is the actual per-tenant secret store?** ADR-0002, 0003, 0005, and 0009 all assume "the secret store keyed by tenant" exists; none
  designs it. Finder's recommendation: an external secret manager (Vault, a cloud KMS-backed secret manager, or Infisical) addressed by an opaque
  alias in the control-plane `tenants` table, with the application role holding only a scoped read credential.
- **Should ADR-0010's future operator CLI get its own role and audit trail now, or only when it's built?** ADR-0010 concedes the tool "needs the
  highest credentials in the system and therefore its own hardening" but doesn't specify what that hardening is. Finder's recommendation: design
  the scoped non-superuser role and the tool's own structured log now, so the CLI isn't built against the superuser DSN by default later.
- **How does a request state which tenant it's for?** No ADR says whether the tenant comes from a URL path, a subdomain, or solely a verified
  token claim, or whether a future JWT should be tenant-scoped. Finder's recommendation: tenant in the URL path (`/t/{tenant_slug}/...`), with
  the access token's audience bound to that same tenant and checked alongside the membership row — defense in depth across URL, token, and
  membership.
- **Is a wall-clock deadline strategy a follow-up to ADR-0009, or its own ADR?** DB pool sizing, `statement_timeout`, and per-run model/embedding
  timeouts all currently rely on library defaults, with no decision recorded anywhere. Finder's recommendation: fold explicit wall-clock budgets
  into ADR-0009 (or a dedicated ADR) covering `SET LOCAL statement_timeout`, explicit pool sizing, and a `model_settings.timeout` per run.
- **Is docker-compose's single shared network for the starter acceptable, or does it need segmentation?** The api container currently has
  unrestricted outbound access and shares a network with postgres and litellm. Finder's recommendation: keep one network for local/dev
  simplicity, but document an egress proxy/allow-list as a production requirement in docs/deployment.md.
- **Is `/openapi.json` a public, versioned API contract or an internal endpoint to gate in production?** docs/mobile.md treats it as the intended
  client-contract source; nothing decides whether that should hold in production. Finder's recommendation: treat it as a CI-generated build
  artifact for client generation, and set `docs_url=redoc_url=openapi_url=None` outside dev/test.

## Method

Eight lens finders — tenant-isolation, authn-authz, agent-surface, secrets-deploy, data-protection, docs-vs-code, tests-ci, and library-facts —
each read the repository and the installed library/dependency source directly (not from memory) and filed findings independently, without seeing
each other's output. A ninth pass, a completeness critic, read the same material looking specifically for gaps the eight lenses had not covered;
its confirmed follow-ups are tagged `critic-gap` above. Every filed finding — 121 confirmed, 24 rejected, 145 total — then went through one
adversarial verification pass that checks three things together: whether the cited evidence is factually correct, whether the described path is
exploitable in a realistic deployment (not one that already assumes host or process compromise), and whether the assigned severity and
recommended fix are calibrated. A finding is refuted if *any one* of those three checks fails, so most of the 24 rejected findings — 20 of them —
had their facts confirmed and were downgraded only on exploitability, severity, or because a draft ADR already answers the question; only 4 were
refuted for an actually incorrect citation or claim. A further 104 explicit checks ("verified as fine") and 15 completeness-critic gaps were
produced alongside the findings above. Finder passes ran on this session's own model; the verification pass ran on Sonnet. In total, 169 agents
(finders, the critic, and per-finding verifiers) contributed to this review. The main limitation worth flagging: because verification is one
combined pass rather than three independent ones, a finding whose facts are entirely correct can still be discarded for a severity or framing
disagreement — which is why this report keeps the "refuted on severity, facts hold" findings visible above rather than silently dropping them.

