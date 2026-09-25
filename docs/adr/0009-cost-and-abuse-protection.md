# ADR-0009: Budgets and rate limits at the gateway per tenant, run limits and a model allow-list in the application

- **Status:** accepted
- **Date:** 2026-09-12
- **Deciders:** JulesCyb
- **Skill version:** ai-app-blueprints v2.0.0 (research 2026-08)

## Context

Verified: the agent endpoints have no rate limit; agent runs use PydanticAI's defaults (50 model
requests, unlimited tool calls per run); one process-wide gateway key serves every tenant, so the
documented "virtual keys with per-tenant budgets" do not exist in code; the LiteLLM container has
no database of its own and inherits the application's `DATABASE_URL` through `env_file`, so the
documented virtual-key setup cannot run as shipped; `get_model()` accepts any string, including
`test` or a foreign provider. ADR-0008 needs per-residency routes and therefore per-tenant
gateway credentials anyway.

## Options

1. **Everything at the gateway** — LiteLLM mandatory, own database, virtual key per tenant with
   budget and rate limit; the application resolves the key per request.
2. **Everything in the application** — quota table under RLS, rate-limit middleware, providers
   called directly; usage accounted from response fields; a distributed limiter to build.
3. **Gateway for hard limits, application for the run** — option 1 plus run limits, a model
   allow-list per residency, and a simple per-membership limit at the agent endpoints.

## Decision

We choose **option 3**.

- The gateway is a required service, not an optional profile. It gets its own database and
  role with no access to the application database, and receives only the variables it needs,
  never the whole `.env`.
- Provisioning a tenant mints a gateway credential with that tenant's budget, rate limit, and
  residency aliases; the credential lives in the secret store keyed by tenant and is resolved
  per request. Model and embedding clients are constructed per tenant, and any cache of them is
  keyed by tenant.
- Every agent run passes run limits (order of ten model requests and twenty tool calls, token
  ceilings set per deployment); exceeding them ends the run with a clear error, never a retry.
- Model names come from an allow-list per residency. `tenants.settings["model"]` may only pick
  from that list; anything else is rejected in the application before it reaches the gateway,
  and the gateway's own allow-list is the second layer.
- Run limits include wall-clock: a deadline per model and embedding call (`model_settings.timeout`),
  `SET LOCAL statement_timeout` inside every tenant transaction, and explicit pool sizing, so a
  stalled provider or query cannot hold a run, a connection, or a streaming client for the
  library defaults of many minutes.
- A per-membership request limit at the agent endpoints catches scripted hammering before the
  gateway budget does; the gateway budget is what finally stops a tenant, and it stops only
  that tenant.

## Consequences

- Positive: cost isolation per tenant becomes real and testable; residency routing and budgets
  share one mechanism; loops are cut short inside the application.
- Negative / costs: an operational dependency on the gateway and its database; tenant
  provisioning gains a credential-minting step; the allow-list must be maintained per
  residency; run limits need tuning per agent.
- What becomes harder later: replacing the gateway means re-implementing budgets and residency
  routing elsewhere (it speaks the OpenAI-compatible protocol, so the client side stays).

## Revisit when …

- the gateway becomes a latency or availability bottleneck (then a gateway per residency)
- budgets must be enforced per membership rather than per tenant
- a provider offers first-party per-project budgets and residency controls good enough to drop
  the gateway
