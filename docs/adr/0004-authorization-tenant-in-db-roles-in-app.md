# ADR-0004: Tenant isolation in the database, roles in the application, documents visible tenant-wide

- **Status:** proposed
- **Date:** 2026-09-12
- **Deciders:** JulesCyb
- **Skill version:** ai-app-blueprints v2.0.0 (research 2026-08)

## Context

Row-Level Security separates tenants. Inside a tenant the starter has no authorization at all:
every member sees every document, `users.role` and the roles in the request context are two
unrelated concepts, `require_role` is never called, and `app.user_id` is set on every
transaction without any policy reading it. A template that leaves this open invites every
derived project to invent its own model.

## Options

1. **Tenant-wide visibility, roles enforced in the application** — RLS stays a pure tenant
   boundary; roles gate actions (invite, change settings, delete) at tools and routes.
2. **Per-document visibility enforced in the database** — owner and shares as a second policy on
   `app.user_id`, plus an "admin sees all" policy. Strongest, but policies combine with OR and
   every new table needs two of them.
3. **Per-document visibility enforced in repositories only** — the developer discipline that RLS
   was introduced to replace.

## Decision

We choose **option 1** for the template.

- Exactly three roles on a membership: `admin`, `member`, `support` (see `CONTEXT.md`). Roles
  decide what a member can do, never what it can see.
- Roles are checked in the application with `RequestContext.require_role` at tools and routes; a
  failed check answers 403, never 500.
- `app.user_id` stays set per transaction and gets a real consumer: audit columns such as
  `created_by` default to it. It is the seam for option 2 if a derived project needs it.
- Option 3 is rejected outright.

## Consequences

- Positive: one policy per table stays the rule; the role model is small enough to test
  exhaustively; audit columns come for free.
- Negative / costs: a derived project with confidential per-member documents must add option 2
  itself; role checks are only as complete as the tools that call them, so every tool needs a
  test for the role it requires.
- What becomes harder later: none of substance; option 2 is additive.

## Revisit when …

- a derived project needs documents that not every member may see (then option 2, as an ADR in
  that project)
- roles need to vary per tenant (then a role table instead of the fixed three)
