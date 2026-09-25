"""agent_credentials.identity_id must be an agent membership of the same tenant (review of #46).

Revision ID: 0041
Revises: 0040
Create Date: 2026-09-25

**The gap this closes.** `issue_agent_credential` (`app/tools/agent_identities.py`) checked only
that the *caller* held the `admin` role, never that `identity_id` was actually an agent identity
belonging to `ctx.tenant_id`. `AgentCredentialRepository.create` (Spec 6 / #45) now checks this
itself, in the same transaction, before the `INSERT` (`app/repositories/agent_credentials.py`).
This migration adds the same invariant one layer down, in the database, so it holds even if a
future change to that repository -- or any other future writer of `agent_credentials`, including
one a later agent adds without reading that repository's docstring -- ever bypasses the
application-level check.

**Why a trigger, not a plain FK.** The natural-looking fix, a composite foreign key from
`agent_credentials (tenant_id, identity_id)` to `memberships (tenant_id, identity_id)`, cannot
express "and that membership's role is `agent`": Postgres foreign keys must reference a `UNIQUE`
or `PRIMARY KEY` constraint, never a partial or filtered one, so there is no way to declare a FK
that only matches rows where `role = 'agent'` -- it would just as happily accept a `member` or
`support` identity's membership, which is exactly the gap already closed at the application layer
and not worth re-opening here. A `BEFORE INSERT OR UPDATE OF tenant_id, identity_id` trigger that
re-runs the same check the repository already runs -- `EXISTS (SELECT 1 FROM memberships WHERE
tenant_id = NEW.tenant_id AND identity_id = NEW.identity_id AND role = 'agent')` -- says exactly
what is meant, needs no new grant (`app` already has `SELECT` on `memberships`, migration 0009),
and needs no `SECURITY DEFINER`: it runs as whichever role performed the write, under that
transaction's own `app.tenant_id`, exactly like every other statement in it.

The trigger only re-checks on `INSERT` or on an `UPDATE` that actually changes `tenant_id` or
`identity_id` -- `revoke()` (sets `revoked_at`) and `verify_and_touch()` (sets `last_used_at`)
never touch either column, so this adds no work to either of those paths.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0041"
down_revision: str | None = "0040"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE FUNCTION agent_credentials_require_agent_membership() RETURNS trigger AS $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM memberships
                WHERE tenant_id = NEW.tenant_id
                  AND identity_id = NEW.identity_id
                  AND role = 'agent'
            ) THEN
                RAISE EXCEPTION
                    'agent_credentials.identity_id (%) has no agent membership in tenant_id (%)',
                    NEW.identity_id, NEW.tenant_id
                    USING ERRCODE = 'foreign_key_violation';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        """
        CREATE TRIGGER agent_credentials_require_agent_membership_trigger
            BEFORE INSERT OR UPDATE OF tenant_id, identity_id ON agent_credentials
            FOR EACH ROW
            EXECUTE FUNCTION agent_credentials_require_agent_membership()
        """
    )


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS agent_credentials_require_agent_membership_trigger "
        "ON agent_credentials"
    )
    op.execute("DROP FUNCTION IF EXISTS agent_credentials_require_agent_membership()")
