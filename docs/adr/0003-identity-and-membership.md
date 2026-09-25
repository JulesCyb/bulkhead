# ADR-0003: Global identities with per-tenant memberships

- **Status:** proposed
- **Date:** 2026-09-12
- **Deciders:** JulesCyb
- **Skill version:** ai-app-blueprints v2.0.0 (research 2026-08)

## Context

A tenant is a customer organization (see `CONTEXT.md`). The starter's `users` table binds a
person to exactly one tenant (`UNIQUE (tenant_id, email)`), and the request context takes
`user_id` as an asserted UUID that is never checked against that table. The operator's own staff
will need to look into a customer's tenant for support, consultants may work for two customers,
and customers may bring their own identity provider — that last point is still open.

## Options

1. **Per-tenant users** (status quo) — one user row per tenant, the tenant follows from the
   login. Simplest schema; a person in two tenants needs two accounts; support access becomes a
   hack; customer-owned identity providers require creating accounts on their behalf.
2. **Global identity plus memberships** — an identity is the pair issuer + subject at the
   identity provider; a membership links identity, tenant, and role. The tenant is chosen per
   request and verified against a membership; the token alone never selects it.
3. **Identity provider owns everything** — tenant and roles as token claims, no membership table.
   No lookup per request, but every tenant change is an identity-provider change, and
   customer-owned providers cannot be trusted to assert *our* tenant ids.

## Decision

We choose **option 2**.

- `identities` is a control-plane table (issuer, subject, display data) without `tenant_id`.
  Rule 2 of `CLAUDE.md` ("every table has tenant_id") gets an explicit exception class,
  *control-plane tables*, with narrow grants: the `app` role may look an identity up by
  issuer + subject and nothing else.
- `memberships(tenant_id, identity_id, role)` replaces `users` and lives under the usual RLS
  policy. The membership lookup runs inside the requested tenant's context, so it needs no RLS
  bypass: no membership row means 403.
- The request context carries `tenant_id`, `identity_id`, and the roles of the membership.
  `AUTH_MODE=jwt` builds it in this order: verify the token against the issuer configured for
  the requested tenant, resolve the identity, resolve the membership, reject otherwise.
- Support access by operator staff is a membership with an explicit role, never a shared
  account and never a bypass; every such membership is visible to the tenant's admins.
- Whether customers bring their own identity provider stays open. The model supports it
  through a per-tenant issuer setting in the control plane; until decided, one operator-run
  provider serves all tenants.

## Consequences

- Positive: one person, one login, any number of tenants; support and consultants are ordinary
  memberships with audit; customer-owned identity providers plug in per tenant.
- Negative / costs: one more control-plane table and a tenant-selection step after login; the
  `users` table and `scripts/seed.py` are rebuilt; the identity lookup is a per-request query
  (cacheable by issuer + subject, key includes nothing tenant-specific by design).
- What becomes harder later: merging two identities of the same person (two providers) needs a
  deliberate tool; moving to option 3 would drop the membership table and its audit trail.

## Revisit when …

- the first customer requires their own identity provider (then decide the per-tenant issuer
  mapping and token validation for multiple issuers)
- agents need identities of their own that are not tied to a person (see the agent-identity
  decision)
- the per-request membership lookup shows up in latency measurements
