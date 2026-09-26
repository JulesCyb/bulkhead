# ADR-0007: Writing tools run only after an approval by the asking member or under a standing grant

- **Status:** accepted
- **Date:** 2026-09-12
- **Deciders:** JulesCyb
- **Skill version:** ai-app-blueprints v2.0.0 (research 2026-08)

## Context

Rule 4 of `CLAUDE.md` demands a confirmation step before the first writing tool and ships none.
pydantic-ai-slim 2.31.0 provides the mechanism (`requires_approval=True` on a tool,
`DeferredToolRequests` as output type, approval streaming in the Vercel adapter with
`sdk_version=6`), and ADR-0006 gives it a trustworthy history to resume from. What is missing is
the policy: who approves, what the approval is bound to, and what autonomous agents may do.
Tool results are prompt-injection surface: a poisoned document can make the agent propose a
writing action, so the approval is the security boundary, not a convenience.

## Options

1. **Per-call approval by the asking member**, verified against a server-side pending action;
   autonomous agents may write only under an explicit standing grant per tool.
2. **Four-eyes** — a tenant admin approves asynchronously from a queue.
3. **Standing approvals for members** — "always allow this tool for me" after the first approval.

## Decision

We choose **option 1**. Option 3 is rejected for the template and derived projects must not add
it: it is the single click that turns prompt injection into a writing action. Option 2 stays a
per-tenant extension if a customer demands it.

- Before asking, the server stores a **pending action**: tenant, conversation, tool, a hash of
  the arguments, the asking membership, an expiry of minutes. The approval is verified against
  that record, never against the message the client sends back. A mismatching hash or an expired
  record fails closed.
- The writing tool re-checks the role of the membership at execution time.
- Every approval, refusal, and execution is an audit record naming actor and means (ADR-0005).
- **Agent identities** cannot call approval-required tools unless a tenant admin has created a
  standing grant for that agent identity and that tool. Under a standing grant the tool runs
  without a person, with the grant recorded in the audit entry.
- `/agents/assistant/run` and `/agents/assistant/stream` cannot carry approvals. They stay as
  one-shot endpoints bound to an agent that has reading tools only; every agent with writing
  tools is reachable through `/api/chat` (or a later jobs API) only.

## Consequences

- Positive: every write has a named person or an admin's explicit grant behind it; the approval
  is bound to what the person actually saw; the mechanism is the library's, not a custom one.
- Positive (spec A3 / #109): the reading/writing split -- which agent a run binds, and the
  read/clear/execute/record sequence around a writing tool's body -- lives in one place,
  `app/agents/run.py`, with the `writing_tool` decorator it re-exports
  (`app/agents/writing_tools.py`); a second writing tool applies that decorator instead of
  copying `rename_document`'s former hand-written wrapper.
- Negative / costs: a pending-action table (tenant-scoped, RLS) and a standing-grant table in
  the tenant; two agents instead of one; every writing tool needs a test that the approval
  binding and the role check hold.
- What becomes harder later: a member cannot make writes frictionless; a batch of writes needs
  one approval per action or a deliberately designed batch tool that is approved as a whole.

Pending actions, standing grants, and audit events are ordinary tenant tables (`tenant_id`,
RLS, registered in `app/db/tenant_tables.py`) and nothing more: when a tenant is erased they are
removed with it, like every other tenant table. The actual erasure mechanics and backup horizon
belong to Spec 9, not to this ADR.

## Revisit when …

- a customer requires four-eyes approval (then option 2 as a tenant setting)
- long-running jobs need approvals outside a conversation (then a jobs API with its own
  approval surface, still bound to pending actions)
