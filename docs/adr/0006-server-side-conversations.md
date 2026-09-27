# ADR-0006: Conversations live on the server, the client contributes only new messages

- **Status:** accepted
- **Date:** 2026-09-12
- **Deciders:** JulesCyb
- **Skill version:** ai-app-blueprints v2.0.0 (research 2026-08)

## Context

`POST /api/chat` runs the PydanticAI Vercel adapter without a server-side `message_history`, so
the whole conversation comes from the request body, including earlier assistant turns and tool
results. Verified against pydantic-ai-slim 2.31.0: a client-supplied `tool-search_documents`
part with a fabricated result reaches the model verbatim. The same library's approval mechanism
(`requires_approval=True`, `DeferredToolRequests`, Vercel `sdk_version=6`) takes the approved
tool call's arguments from that history on resume. A trustworthy confirmation step for writing
tools (rule 4 of `CLAUDE.md`), an audit trail, continuing a conversation on another device
(`docs/mobile.md`), and any agent memory all need one trusted record of the conversation.

## Options

1. **Stateless** (status quo) — the client holds the history. No storage, no retention; history
   is forgeable, approvals cannot trust their own input, no device hand-over.
2. **Server-side under RLS** — a conversation table with `tenant_id`; the server loads the
   history and takes only the newest user message from the client.
3. **Stateless but signed** — the server signs assistant and tool messages, the client replays
   them. Integrity without storage, but no audit, no hand-over, cryptography in the chat protocol.

## Decision

We choose **option 2**.

- Conversations and their messages are tenant tables under the standard RLS policy, keyed by
  the client's conversation id, with `created_by` from `app.user_id` (ADR-0004).
- On every chat request the server loads the history from the repository, passes it as
  `message_history`, and uses only the last user message from the body; assistant and tool
  parts sent by the client are dropped, not merged. New messages produced by the run are
  appended by the server.
- A retention period is a tenant setting (`tenants.settings["retention_days"]`,
  `app/tenant_settings.py`) with a safe default of 90 days (`DEFAULT_RETENTION_DAYS`) when a
  tenant never sets its own; expired conversations are deleted by a job
  (`app/retention.py`, run via `scripts/retention.py`) that runs per tenant, through the same
  `tenant_session(ctx)` every other request uses — never a superuser or bypass-RLS statement.
  Deleting a tenant deletes its conversations for free, via the existing cascade.
- A tenant may shorten or lengthen its own retention period, but never past a deployment-wide
  maximum of 365 days (`Settings.max_retention_days`, `MAX_RETENTION_DAYS`, #84) — GDPR
  Art. 5(1)(e)'s storage-limitation principle requires personal data be kept "no longer than is
  necessary," and an unbounded tenant-chosen period would defeat that regardless of how
  conservative the 90-day default is. The maximum is enforced twice: on write, wherever
  `retention_days` is set (`app.operator.create`'s `_validate_retention_days`) rejects a value
  above it before anything is written; on read, `app.tenant_settings.effective_retention_days`
  clamps a pre-existing stored value above the current maximum down to it — the fail-safe for a
  row written under an earlier, higher cap.
- This is the prerequisite for the approval flow (next decision) and for audit records.

## Consequences

- Positive: history is trustworthy; approvals, audit, device hand-over, and memory have one
  home; the stateless protocol on the wire stays unchanged for the frontend.
- Negative / costs: a migration, a repository, a retention job; conversations are personal data
  of every tenant in one table (pooled tier), so retention and deletion are not optional; the
  `/agents/assistant/*` endpoints either gain a conversation id or stay one-shot without history.
- What becomes harder later: none of substance; option 3 could be layered on for offline
  clients if ever needed.

## Revisit when …

- a tenant requires conversations never to be stored (then option 3 for that tenant, with
  writing tools disabled)
- conversation volume makes retention jobs or storage a cost problem
