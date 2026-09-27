# ADR-0012: A request names its tenant in the URL path, and path, token audience, and membership must agree

- **Status:** accepted
- **Date:** 2026-09-12
- **Deciders:** JulesCyb
- **Skill version:** ai-app-blueprints v2.0.0 (research 2026-08)

## Context

Today the tenant comes from a bare `X-Tenant-Id` header bound to nothing. ADR-0003 says the
tenant is chosen per request and verified against a membership, but not how a request states
it. CORS is one process-wide origin list, `docs/mobile.md` wants a frozen `/v1/` contract, and
ADR-0005 needs an OAuth resource per tenant for MCP.

## Options

1. **URL path** — `/v1/t/{tenant_id}/…`; the access token is issued for that tenant (audience);
   the server checks that path, audience, and membership agree.
2. **Subdomain per tenant** — strongest origin-level separation for cookies and CORS; wildcard
   TLS, local-development friction, and mobile and MCP clients still need the name.
3. **Token claim only** — the identity provider asserts `tenant_id`; simplest client, but a
   customer-owned identity provider cannot be trusted to assert the operator's tenant ids, and a
   person in two tenants re-authenticates on every switch.

## Decision

We choose **option 1**.

- Every tenant-scoped route lives under `/v1/t/{tenant_id}/`. The tenant id in the path is the
  request's statement of intent; nothing else may name the tenant.
- The access token carries the tenant as its audience. A token for tenant A is rejected on a
  path for tenant B. Switching tenants means a new token, not a new login.
- `get_context` resolves the identity from the token, checks the audience against the path,
  then loads the membership inside that tenant's context. All three must agree; a mismatch is
  403 and a security event.
- For MCP over HTTP, the tenant's path prefix is the OAuth resource; a connection is bound to
  one tenant.
- `AUTH_MODE=dev-headers` keeps supplying only what the token would supply (identity and
  roles); the tenant still comes from the path, so development and production share the
  routing code.
- Two tabs for two tenants are two URLs; a re-login in one cannot redirect the other.

## Consequences

- Positive: the tenant is visible in logs, contracts, and clients; the three-way check makes a
  forged header or a replayed token useless; CORS can later grow per tenant without changing
  the routing.
- Negative / costs: every route and every client gains a path segment; the identity provider
  must issue tenant-scoped tokens (audience) or the application must exchange a login token for
  a tenant token itself; the seed and the frontend example change.
- What becomes harder later: moving to subdomains keeps the path and adds an origin check, so
  nothing is lost.

## Revisit when …

- a customer requires cookie sessions on an origin of their own (then option 2 on top)
- the identity provider cannot issue per-tenant audiences (then an application-side token
  exchange, still bound to the path)
