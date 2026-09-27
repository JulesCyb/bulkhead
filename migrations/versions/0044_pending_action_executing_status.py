"""Add `executing` to `pending_actions.status`: claim before a writing tool runs (ADR-0007, #122).

Revision ID: 0044
Revises: 0043
Create Date: 2026-09-27

Migration 0043 gave a pending action its whole lifecycle but left one gap between two
transactions: the approval validator (`app/tools/approvals.py`'s `require_approval`) verified an
`approved` action and committed, the tool body ran, and only afterwards did the outcome path move
the row `approved -> executed | execution_failed`. For that whole window the row was still
`approved`, so two resumes of the same approval sent concurrently both passed verification and
both executed (reproduced during #122's triage: `requested, approved, executed, executed`).

The fix is a claim, made atomically with the verification: in the same transaction that verifies
the action, `PendingActionRepository.claim_for_execution()` moves it `approved -> executing` with
a guarded `UPDATE ... WHERE status = 'approved'`, and only the caller whose update moved the row
may run the tool; a concurrent resume updates zero rows and is refused (`status_executing`). The
outcome then moves `executing -> executed | execution_failed`, never from `approved`. That needs
one more stored state, so the full vocabulary becomes `pending`, `approved`, `refused`, `expired`,
`executing`, `executed`, `execution_failed`. Every transition still lives in
`PendingActionRepository` (`app/repositories/pending_actions.py`); nothing outside it writes
`status`.

`executing` is non-terminal but only the outcome path leaves it: the sweep
(`app/pending_action_sweep.py`) touches only `pending` rows, and a row stuck in `executing` after a
process crash is a visible, fail-closed state (it can never run again), deliberately not recovered
by any timeout -- that would be a new decision under ADR-0007, not part of this migration.

Nothing else about the table changes: RLS (`ENABLE`/`FORCE`, the tenant-isolation policy), the
`app` role's `SELECT, INSERT, UPDATE` grant (0022 -- the `UPDATE` already covers the claim), and
the tenant-table registry (`app/db/tenant_tables.py`) are untouched. The constraint is dropped and
re-added under the same name, `pending_actions_status_check`, exactly as 0043 did.

**Downgrade limitation.** 0043's six-value constraint cannot be restored while any row carries
`executing` -- there is no truthful way to say whether such an action ran. The downgrade therefore
refuses (raises "cannot downgrade 0044", leaving the schema at this revision) when such rows
exist, rather than silently rewriting audit-relevant state. As in 0043, the refusal comes from
re-adding the narrower constraint, whose validation sees every row -- not from a `SELECT`
pre-check, which under FORCE ROW LEVEL SECURITY and no tenant context would see none (see
`downgrade()`).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0044"
down_revision: str | None = "0043"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE pending_actions DROP CONSTRAINT pending_actions_status_check")
    op.execute(
        """
        ALTER TABLE pending_actions ADD CONSTRAINT pending_actions_status_check
            CHECK (status IN (
                'pending', 'approved', 'refused', 'expired',
                'executing', 'executed', 'execution_failed'
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
                CHECK (status IN (
                    'pending', 'approved', 'refused', 'expired', 'executed', 'execution_failed'
                ));
        EXCEPTION WHEN check_violation THEN
            RAISE EXCEPTION 'cannot downgrade 0044: pending_actions rows carry %',
                'executing; decide what they should become first';
        END $$;
        """
    )
