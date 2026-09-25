# ADR-0008: Residency is a tenant setting, and every content-bearing path honours it

- **Status:** proposed
- **Date:** 2026-09-12
- **Deciders:** JulesCyb
- **Skill version:** ai-app-blueprints v2.0.0 (research 2026-08)

## Context

The README claims "GDPR via EU regions + DPA". Verified against the code and the installed
libraries: every search query is embedded at OpenAI in the US regardless of the model route
(`app/embeddings.py` defaults to api.openai.com); `logfire.instrument_pydantic_ai()` exports
prompts, tool results, and document snippets by default (`include_content=True`) into one
Langfuse project for all tenants, at any `LANGFUSE_HOST`; the Docker image never installs the
observability extra, so tracing is silently absent in the shipped container. The model route
itself is Claude by default (`anthropic:claude-sonnet-4-5`, a stale id); embeddings go elsewhere
because Anthropic offers no embedding model (its documentation points to Voyage AI).

## Options

1. **Hard EU guarantee, fail-closed** — one region for everyone, enforced at startup by an
   allow-list of endpoints; tracing without content.
2. **Residency per tenant** — a tenant setting selects the region; model, embeddings, and
   tracing routes are resolved from it per request; option 1's guard becomes the baseline.
3. **No guarantee** — document the processor chain, keep the defaults, drop the README claim.

## Decision

We choose **option 2**, with option 1's guard as the baseline.

- `tenants.settings["residency"]` names the residency (for example `eu`). Every path that
  carries content resolves its route from it: the model (a gateway alias or provider client per
  residency), the embedding provider, and the tracing sink. A request never uses a route of
  another residency; an unresolvable route fails closed.
- At startup the process validates every configured endpoint against an allow-list per
  residency and refuses to start otherwise. There is no default embedding provider; the value
  must be set explicitly.
- Tracing exports identifiers only (`include_content=False`); content in traces is a tenant
  opt-in, and the trace sink is chosen per residency. The observability extra becomes a regular
  dependency so the shipped image can actually trace.
- The MCP path hands snippets to the model of the connecting client; that lies outside the
  operator's processor chain and is documented as the customer's responsibility, with the
  snippet cap kept.
- Per-tenant routing needs per-tenant gateway credentials (next decision, ADR-0009).

## Consequences

- Positive: the README claim becomes true and testable; a tenant with a different residency
  needs a setting, not a deployment; tracing stops being a silent content export.
- Negative / costs: one embedding model family across residencies, because the vector dimension
  is fixed per column (1536 today) and pgvector indexes need a fixed dimension; a residency
  therefore selects the *provider region*, never a different embedding model; changing the
  family is a migration and a re-embedding. Provider allow-lists must be maintained.
- What becomes harder later: adding a residency means an allow-list entry, a gateway alias, an
  embedding endpoint, and a trace sink; none of it is code, all of it must be tested.

## Revisit when …

- a tenant needs a residency that no available embedding provider serves with the chosen model
  family (then a second family and a second column, per tenant)
- the model provider offers a first-party residency control that makes gateway aliases per
  region unnecessary
