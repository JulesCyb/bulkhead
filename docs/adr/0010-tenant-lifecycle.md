# ADR-0010: Tenant lifecycle is an operator tool now, a control-plane API with the second customer

- **Status:** accepted
- **Date:** 2026-09-12
- **Deciders:** JulesCyb
- **Skill version:** ai-app-blueprints v2.0.0 (research 2026-08)

## Context

The only lifecycle code is `scripts/seed.py`, which creates a tenant and a user with the owner
DSN. There is no suspension, no deletion path, no statement about backups or traces. After
ADR-0002 to ADR-0009 a tenant consists of: a control-plane record with isolation tier and
residency, possibly a dedicated database, a gateway credential with budget, memberships,
conversations, traces, and copies in backups.

## Options

1. **Operator command-line tool only** — `create`, `suspend`, `delete`, run with the owner role
   and the gateway master credential; idempotent, tested, documented; no API surface.
2. **Control-plane API with a platform-admin principal** — its own principal type above tenants
   and its own database role; enables self-service and billing at the price of a second
   authentication system in the template.
3. **Option 1 now, option 2 with the second customer.**

## Decision

We choose **option 3**, and option 1 carries two rules:

- **Suspension precedes erasure.** A suspended tenant is a control-plane state that the context
  resolution checks on every request, so a cancelled customer is out immediately while the
  erasure deadline runs. Suspension deletes nothing and is reversible.
- **Erasure is complete and recorded.** The tool removes the tenant from every place its data
  lives (database rows via the cascade, a dedicated database if any, the gateway credential,
  conversations, traces by tenant) and writes a record of what was removed where. The backup
  horizon is documented as the erasure deadline because it holds the last copy; backups are
  never restored for an erased tenant.
- Trace deletion per tenant requires `tenant_id` as a flat span attribute on every span, not a
  JSON string on the run span; this is part of ADR-0008's tracing changes.
- The tool replaces `scripts/seed.py`; `create` mints the gateway credential (ADR-0009),
  writes isolation tier and residency (ADR-0002, ADR-0008), and creates the first admin
  membership (ADR-0003).

## Consequences

- Positive: a customer can be provisioned and removed by one documented command each; the
  erasure record answers the auditor's question; suspension gives a safe first step.
- Negative / costs: every new place that stores tenant data must be added to the tool, and a
  test must fail when a tenant table is not covered by erasure; the tool needs the highest
  credentials in the system and therefore its own hardening.
- What becomes harder later: none; option 2 wraps the same operations in an API.

## Revisit when …

- the second customer arrives (then option 2, with billing)
- a customer needs self-service member management before that
