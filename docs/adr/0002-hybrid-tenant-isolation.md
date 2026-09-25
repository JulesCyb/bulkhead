# ADR-0002: Hybrid tenant isolation — pooled by default, dedicated database on demand

- **Status:** accepted (decided on judgement, no contractual demand yet — see "Revisit when")
- **Date:** 2026-09-12
- **Deciders:** JulesCyb
- **Skill version:** ai-app-blueprints v2.0.0 (research 2026-08)

## Context

The starter isolates tenants with Row-Level Security inside one shared Postgres. The owner's goal
for the template includes "customers with their own users and databases". No customer demands a
dedicated database contractually today.

RLS in a shared database protects against developer mistakes (a query without a tenant filter),
not against a compromised application process: the `app` role sets `app.tenant_id` itself. A
forgotten policy on a new table exposes every tenant at once. Some customers, regulators, and
data-residency requirements will not accept logical separation alone.

## Options

1. **Pooled only** (status quo) — one database, RLS separates. Cheapest to run and migrate; one
   policy bug hits all customers; no answer for a customer who demands physical separation.
2. **Schema per tenant** — same database, one schema each. Little gain over 1 (same instance, same
   credentials), migrations × N.
3. **Database per tenant** — hard separation, own backups, own key and region possible; migrations
   and connection pools × N, a control plane outside RLS is required, expensive with many small
   tenants.
4. **Hybrid** — 1 as the default isolation tier, 3 for tenants that need it; the session layer
   resolves the database from the control plane.

## Decision

We choose **option 4**, built in two steps:

1. **Now, the seam only.** `tenant_session(ctx)` resolves the engine from the control plane
   (`tenants.isolation_tier` plus a database alias). Credentials for dedicated databases live in
   configuration or a secret store keyed by that alias — never in tenant-editable settings. RLS
   stays enabled and forced in every database, pooled or dedicated, as the second layer.
   Migrations run once per database alias. The RLS integration test also covers routing: a
   dedicated tenant's data must not be reachable through the pooled engine.
2. **The first dedicated tenant only when a trigger below fires.** Until then the template ships
   with one pooled database and the seam is exercised by tests, not by customers.

## Consequences

- Positive: the "own database" promise can be kept without rebuilding the data layer; pooled
  stays cheap for small tenants; residency per tenant becomes possible.
- Negative / costs: migrations and pools per database alias; the control plane becomes the most
  sensitive data in the system; provisioning a dedicated tenant is an operator action, not a
  self-service one.
- What becomes harder later: moving a tenant between tiers needs a data-migration tool; any
  cross-tenant analytics must be aggregated across databases.

## Revisit when …

- a customer demands a dedicated database or a specific region contractually
- regulated data (health, finance, public sector) enters a tenant
- one tenant dominates load or size and degrades the others
- a customer needs their own backup, restore, or export cadence
- conversely: if no trigger has fired after the first few customers, consider dropping to
  option 1 and removing the seam
