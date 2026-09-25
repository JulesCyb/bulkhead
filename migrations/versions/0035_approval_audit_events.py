"""Audit trail for the approval mechanism (ADR-0007, Spec 5 / #39).

Revision ID: 0035
Revises: 0032
Create Date: 2026-09-25

Every milestone the approval mechanism can reach -- a write requested, approved, refused,
expired, denied for lack of a grant, executed, or failed to execute -- becomes one append-only
row in `approval_audit_events`, naming the acting membership and, where applicable, the specific
pending action (0022) or standing grant (0031) behind it. This is deliberately a *second* kind of
audit record, distinct from the "who last touched this row" audit columns 0010 established for
`documents`: 0010 answers "who wrote this row"; this table answers "who authorized this write,
and how" (ADR-0007, CLAUDE.md rule 4).

`approval_audit_events` follows the standard tenant-table shape (CLAUDE.md rule 2, same as
`pending_actions`/`standing_grants`): a required, indexed `tenant_id NOT NULL REFERENCES
tenants(id)`, `ENABLE`/`FORCE ROW LEVEL SECURITY`, and the usual
`tenant_id = current_setting('app.tenant_id', true)::uuid` policy (USING and WITH CHECK) -- so an
audit record written for one tenant is simply not a row a second tenant's session can see, by
construction, exactly as the ticket's acceptance criteria require.

`actor_membership_id` references `memberships (id)`, not `control.identities`: the same split
`pending_actions` and `standing_grants` already draw, for the same reason -- the record is
inherently about a membership's role at the moment the milestone happened, not about the global
identity.

`kind` is restricted by a `CHECK` constraint to the exact seven milestones the ticket names:
`requested`, `approved`, `refused`, `expired`, `denied_for_lack_of_grant`, `executed`, and
`failed_to_execute` -- the vocabulary this table itself models, not an open string a caller could
drift.

`pending_action_id` and `standing_grant_id` are both nullable, plain `uuid` columns -- no foreign
key, deliberately, following the precedent `control.operator_actions`/`control.tenant_erasures`
(migration 0004) already set for an audit row that must document, and outlive, the record it
names: a pending action is cascade-deleted with its conversation (0022), and nothing about this
audit table's own job -- "was this authorized, and how" -- should ever be allowed to disappear
because the pending action or standing grant it names later does. Exactly one of the two is set
for a milestone that came from a person's approval (`pending_action_id`) or an agent identity's
standing grant (`standing_grant_id`); a milestone with no specific record behind it yet (e.g.
`denied_for_lack_of_grant`, where by definition no grant exists) leaves both null. Ordinary
(non-unique, non-partial) indexes on both columns make "every milestone for this one pending
action/grant, most recent first" (the read this ticket's last acceptance criterion asks for) a
direct index lookup, not a sequential scan.

`seq` is a `GENERATED ALWAYS AS IDENTITY` column purely for that ordering: Postgres's `now()` is
frozen for the whole transaction, so several milestones written in one transaction (a common case
here -- a single write request can move straight from `requested` to `executed`) would otherwise
share the exact same `created_at` and sort arbitrarily against each other. `seq` always advances,
transaction or no, so "most recent first" (`ORDER BY seq DESC`) is unambiguous even for events
that share a timestamp to the microsecond.

Grants for `app`: `SELECT, INSERT` only -- no `UPDATE`, no `DELETE`. Unlike `pending_actions`
(which allows `UPDATE` to resolve a row) and `standing_grants` (which allows `UPDATE` to revoke
one), this table has no mutation of an existing row anywhere in its own rules: a milestone, once
written, is permanent. The grant list itself enforces "append-only" at the database layer, the
same way `control.operator_actions` (0004) does for the operator tool's own audit trail -- append-
only is a property of what the `app` role (and, in application code,
`app/repositories/approval_audit.py`'s own write surface) can even attempt, not a convention to
remember.
"""

from collections.abc import Sequence

from alembic import op

_TABLES_AT_THIS_REVISION: tuple[str, ...] = ("approval_audit_events",)

revision: str = "0035"
down_revision: str | None = "0032"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE approval_audit_events (
            id                    uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            seq                   bigint GENERATED ALWAYS AS IDENTITY,
            tenant_id             uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            kind                  varchar(30) NOT NULL
                                      CHECK (kind IN (
                                          'requested', 'approved', 'refused', 'expired',
                                          'denied_for_lack_of_grant', 'executed',
                                          'failed_to_execute'
                                      )),
            tool_name             varchar(200) NOT NULL,
            actor_membership_id   uuid NOT NULL REFERENCES memberships (id),
            -- No REFERENCES on purpose: this row must outlive the pending action or standing
            -- grant it names (see module docstring).
            pending_action_id     uuid,
            standing_grant_id     uuid,
            details               jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at            timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX approval_audit_events_tenant_idx ON approval_audit_events (tenant_id)")
    op.execute(
        "CREATE INDEX approval_audit_events_pending_action_idx "
        "ON approval_audit_events (pending_action_id)"
    )
    op.execute(
        "CREATE INDEX approval_audit_events_standing_grant_idx "
        "ON approval_audit_events (standing_grant_id)"
    )

    for table in _TABLES_AT_THIS_REVISION:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"""
            CREATE POLICY {table}_tenant_isolation ON {table}
                USING      (tenant_id = current_setting('app.tenant_id', true)::uuid)
                WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid)
            """
        )

    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app') THEN
                GRANT SELECT, INSERT ON approval_audit_events TO app;
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS approval_audit_events")
