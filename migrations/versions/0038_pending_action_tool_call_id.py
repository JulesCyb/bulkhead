"""Correlate a pending action with the model's own tool call (ADR-0007, Spec 5 / #40).

Revision ID: 0038
Revises: 0035
Create Date: 2026-09-25

The writing-tool approval flow needs to find, on the *resumed* run, the exact pending action a
given deferred tool call created on the *first* run -- pydantic-ai's own resolution of an
approval/refusal is keyed by the model's `tool_call_id` (stable across the two runs because the
unresolved `ToolCallPart` is replayed from the trusted, server-held conversation history, ADR-0006),
never by anything this application invents. `tool_call_id` is that same key, stored on the row
that was written down before the approval was ever shown to a member (migration 0022's own
ordering guarantee), so `app/tools/approvals.py` can look the row back up on the resumed run and
on the request that resolves a member's approve/refuse decision, without trusting a value the
client sends back for it.

Not unique: a tool call id is expected to name at most one row in practice (one proposal per
call), but nothing here depends on that being enforced by the database -- `get_by_tool_call`
takes the most recent match, the same fail-open-to-fail-closed shape the rest of this table
already uses (see `PendingActionRepository.verify()`).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0038"
down_revision: str | None = "0035"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Fresh-install assumption (matches migration 0010's own precedent): pending_actions is empty
    # on every database this template ships against, so NOT NULL with no backfill is safe here.
    op.execute("ALTER TABLE pending_actions ADD COLUMN tool_call_id varchar(200) NOT NULL")
    op.execute(
        "CREATE INDEX pending_actions_tool_call_idx "
        "ON pending_actions (tenant_id, conversation_id, tool_call_id)"
    )
    # No new grant: pending_actions already grants app SELECT/INSERT/UPDATE on the whole row
    # (migration 0022), which covers the new column.


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS pending_actions_tool_call_idx")
    op.execute("ALTER TABLE pending_actions DROP COLUMN IF EXISTS tool_call_id")
