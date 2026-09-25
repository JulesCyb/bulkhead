"""Pending actions for writing-tool approval (ADR-0007, Spec 5 / #37).

Revision ID: 0022
Revises: 0020
Create Date: 2026-09-25

Before a writing-tool approval is ever shown to a member, the server writes down a **pending
action**: which tenant, which conversation, which tool, a hash of the exact arguments, the
asking membership, and an expiry a few minutes out. Later, an approval is checked strictly
against that stored record -- never against whatever the client resends -- so this table is the
security boundary ADR-0007 describes, not a courtesy log.

`pending_actions` gets the same four mandatory parts every table in this project gets (CLAUDE.md
rule 2): `tenant_id NOT NULL REFERENCES tenants(id)`, an index on it, `ENABLE`/`FORCE ROW LEVEL
SECURITY`, and the standard tenant-isolation policy (USING and WITH CHECK on
`current_setting('app.tenant_id', true)`).

`conversation_id` follows the same composite-FK shape `messages` uses (migration 0020):
`FOREIGN KEY (tenant_id, conversation_id) REFERENCES conversations (tenant_id, conversation_id)
ON DELETE CASCADE` -- a pending action cannot outlive the conversation it was raised in, and
deleting a tenant (or the retention job deleting one tenant's expired conversation) removes its
pending actions for free, no separate cleanup step.

`asking_membership_id` and `resolved_by` both reference `memberships (id)` -- the asking/
resolving member is a tenant-scoped fact (Spec 2/3), unlike `created_by` elsewhere in this schema
which names a *global* `control.identities` row. A pending action is inherently about a
membership's role at the moment it was raised (and, later, at the moment it is answered), so the
membership itself -- not the underlying identity -- is what this table names.

`args_hash` is a hex-encoded SHA-256 digest of the tool name and its exact arguments
(`app/repositories/pending_actions.py::hash_arguments`), computed the same way at creation and at
verification time -- a mismatch (a different call than the one that was proposed) refuses
verification outright, never a fallback to "close enough."

`status` is restricted by a `CHECK` constraint to `pending`, `approved`, or `refused` -- the
three states this table itself models. "Expired" and "executed" are not stored states here: an
expired pending action is simply one whose `expires_at` has passed while `status` is still
`pending` (verified against the real clock at verification time, never a separate sweep that
could race it), and "executed" is the writing tool's own outcome, recorded by the audit module a
later ticket in this spec adds -- this table's job ends at "was this approved, and does it still
match."

Grants for `app`: `SELECT, INSERT, UPDATE` -- `UPDATE` (unlike `conversations`/`messages`, which
get none) because resolving a pending action (member approves or refuses) is exactly the one
`UPDATE` this table's own rules allow, transitioning `status` from `pending` to a terminal state.
No `DELETE` grant, matching `agent_credentials` (migration 0021): the only way a row disappears
is the cascading delete from its conversation or tenant, never an application code path.
"""

from collections.abc import Sequence

from alembic import op

_TABLES_AT_THIS_REVISION: tuple[str, ...] = ("pending_actions",)

revision: str = "0022"
down_revision: str | None = "0020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE pending_actions (
            id                    uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id             uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            conversation_id       varchar(200) NOT NULL,
            tool_name             varchar(200) NOT NULL,
            args_hash             varchar(64) NOT NULL,
            asking_membership_id  uuid NOT NULL REFERENCES memberships (id),
            status                varchar(20) NOT NULL DEFAULT 'pending'
                                      CHECK (status IN ('pending', 'approved', 'refused')),
            created_at            timestamptz NOT NULL DEFAULT now(),
            expires_at            timestamptz NOT NULL,
            resolved_at           timestamptz,
            resolved_by           uuid REFERENCES memberships (id),
            FOREIGN KEY (tenant_id, conversation_id)
                REFERENCES conversations (tenant_id, conversation_id) ON DELETE CASCADE
        )
        """
    )
    op.execute("CREATE INDEX pending_actions_tenant_idx ON pending_actions (tenant_id)")

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
                GRANT SELECT, INSERT, UPDATE ON pending_actions TO app;
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS pending_actions")
