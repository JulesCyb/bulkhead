# ADR-0005: Agents act by delegation; autonomous agents get an identity of their own

- **Status:** proposed
- **Date:** 2026-09-12
- **Deciders:** JulesCyb
- **Skill version:** ai-app-blueprints v2.0.0 (research 2026-08)

## Context

"Native agent support" is one of the template's three goals. Today the PydanticAI agent runs in
the request context of whoever called the API, and the MCP server has one process-wide identity
read from two environment variables, over stdio only. The audit trail cannot tell "member X
searched" from "an agent searched on behalf of X", a scheduled job with no person logged in has
no identity to run as, and a customer cannot connect automation of their own.

## Options

1. **Delegation only** — an agent always carries the membership of a person. Clean audit, no
   credentials to manage; but no autonomous jobs and no customer automation.
2. **Agents as identities only** — every agent call runs as an agent identity with its own
   membership. Autonomous work becomes possible, but the person behind an interactive request is
   lost from the audit trail.
3. **Both, with a fixed rule** — interactive use is always delegation; autonomous use always runs
   through an agent identity.

## Decision

We choose **option 3**.

- **Interactive** requests (API, chat, an MCP client used by a person) run by delegation: the
  request context carries the person's membership, and every audit record names the person as
  actor and the agent as means.
- **Autonomous** work (scheduled jobs, a customer's own automation, an MCP client without a
  person) runs as an agent identity: an identity of kind *agent* with a membership of role
  *agent*, reading tools only unless the tenant's admin grants more. Credentials are issued per
  tenant, listed for and revocable by the tenant's admins, and never shared across tenants.
  There is no global key of any kind.
- **MCP** moves to Streamable HTTP with OAuth for anything beyond local development; the tenant
  and identity come from the token of the connection, per connection. The process-wide
  `context_provider` survives only as an explicit development mode guarded the same way as
  `AUTH_MODE=dev-headers`.
- The request context grows a second field for the means (which agent, which credential) so
  audit records can carry both.

## Consequences

- Positive: audit answers "who" and "through what"; autonomous jobs and customer automation are
  first-class; revocation is per tenant and per agent.
- Negative / costs: a credential lifecycle (issue, list, rotate, revoke) per tenant; a fourth
  role; MCP needs an OAuth-capable transport and a client that supports it; two code paths for
  building the context.
- What becomes harder later: collapsing to option 1 would strand every autonomous integration;
  collapsing to option 2 would lose the person from the audit trail.

## Revisit when …

- agents need to delegate to other agents (then decide whether chains of delegation are
  recorded or flattened)
- a tenant wants agent credentials scoped narrower than a role (then per-credential tool scopes)
- the MCP specification changes its authorization model
