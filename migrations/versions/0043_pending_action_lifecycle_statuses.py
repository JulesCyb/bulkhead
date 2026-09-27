"""Widen `pending_actions.status` to the full lifecycle (ADR-0007, #82).

Revision ID: 0043
Revises: 0042
Create Date: 2026-09-27

Migration 0022 restricted `status` to `pending`, `approved`, `refused` and deliberately left
"expired" and "executed" out of the table: an expired action was one whose `expires_at` had
passed while still `pending`, and "executed" lived only in the audit trail (migration 0035). That
left two gaps #82 closes:

- A proposal nobody ever answers never produced an `expired` audit event, because the only place
  one was written was a *late resume* (`app/tools/approvals.py`). The sweep job
  (`app/pending_action_sweep.py`, `scripts/sweep_pending_actions.py`) now marks such a row
  `expired` and writes that event -- and it needs a stored state to make that idempotent: a row
  already `expired` is simply not `pending` any more, so a second sweep finds nothing.
- An approved action that already ran looked exactly like one still waiting to run. It now moves
  to `executed` or `execution_failed`, in the same transaction as the matching audit row, so
  `PendingActionRepository.verify()` -- which requires exactly `approved` -- refuses to verify it
  a second time.

The full vocabulary is therefore `pending`, `approved`, `refused`, `expired`, `executed`,
`execution_failed`. Every transition lives in `PendingActionRepository`
(`app/repositories/pending_actions.py`); nothing outside it writes `status`.

Nothing else about the table changes: RLS (`ENABLE`/`FORCE`, the tenant-isolation policy), the
`app` role's `SELECT, INSERT, UPDATE` grant (0022 -- the `UPDATE` already covers these new
transitions), and the tenant-table registry (`app/db/tenant_tables.py`) are untouched.

The constraint is the inline column `CHECK` 0022 created, which Postgres names
`pending_actions_status_check`; it is dropped and re-added under the same name.

**Downgrade limitation.** The three-value constraint cannot be restored while any row carries one
of the three new values -- there is no truthful way to map `executed` or `expired` back onto
`pending`/`approved`/`refused`. The downgrade therefore refuses (raises "cannot downgrade 0043",
leaving the schema at this revision) when such rows exist, rather than silently rewriting
audit-relevant state; an operator who really wants to go back must decide what those rows should
become first. The refusal comes from re-adding the narrow constraint, whose validation sees every
row -- not from a `SELECT` pre-check, which under FORCE ROW LEVEL SECURITY and no tenant context
would see none (see `downgrade()`).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0043"
down_revision: str | None = "0042"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE pending_actions DROP CONSTRAINT pending_actions_status_check")
    op.execute(
        """
        ALTER TABLE pending_actions ADD CONSTRAINT pending_actions_status_check
            CHECK (status IN (
                'pending', 'approved', 'refused', 'expired', 'executed', 'execution_failed'
            ))
        """
    )


def downgrade() -> None:
    # Not an `IF EXISTS (SELECT ... FROM pending_actions)` pre-check: the migration role
    # (`app_owner`) is subject to FORCE ROW LEVEL SECURITY with no tenant context, so such a check
    # would see zero rows and pass. Constraint validation itself sees every row regardless of RLS;
    # its `check_violation` is caught here only to name the real reason, and the exception block
    # undoes the DROP before re-raising.
    op.execute(
        """
        DO $$
        BEGIN
            ALTER TABLE pending_actions DROP CONSTRAINT pending_actions_status_check;
            ALTER TABLE pending_actions ADD CONSTRAINT pending_actions_status_check
                CHECK (status IN ('pending', 'approved', 'refused'));
        EXCEPTION WHEN check_violation THEN
            RAISE EXCEPTION 'cannot downgrade 0043: pending_actions rows carry %',
                'expired/executed/execution_failed; decide what they should become first';
        END $$;
        """
    )
