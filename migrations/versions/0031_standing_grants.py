"""Standing grants for agent identities (ADR-0005, ADR-0007, Spec 5 / #38).

Revision ID: 0031
Revises: 0022
Create Date: 2026-09-25

An **agent identity** (ADR-0005) may call a writing tool with no person present only when a
tenant admin has explicitly authorized it, tool by tool -- ADR-0007's "standing grant". This
table is the durable record of that authorization: which agent membership may call which tool,
who granted it and when, and -- since a grant is revoked, never deleted -- who revoked it and
when.

`standing_grants` follows the standard tenant-table shape (CLAUDE.md rule 2, same as
`pending_actions` in 0022): a required, indexed `tenant_id NOT NULL REFERENCES tenants(id)`,
`ENABLE`/`FORCE ROW LEVEL SECURITY`, and the usual `tenant_id = current_setting('app.tenant_id',
true)::uuid` policy (USING and WITH CHECK).

`agent_membership_id`, `granted_by`, and `revoked_by` all reference `memberships (id)`, not
`control.identities` -- exactly the split `pending_actions` (0022) already draws and for the same
reason: a standing grant is inherently a tenant-scoped fact about a *membership*'s role at the
moment it was granted (and, later, at the moment it is revoked), not a fact about the global
identity. Whether the target membership actually carries the `agent` role is checked by
`app/repositories/standing_grants.py` before the row is ever written (`memberships_role_valid`
already restricts `role` to the four legal values at the database layer; carrying `agent`
specifically is this feature's own rule, not a `CHECK` constraint on this table, because it must
be re-checked against the membership's *current* role, not the role at grant time).

**At most one active grant per tenant, agent membership, and tool.** Enforced by a partial unique
index on `(agent_membership_id, tool_name) WHERE revoked_at IS NULL` -- `tenant_id` is already
implied by `agent_membership_id` (a membership belongs to exactly one tenant), so this is the
partial-uniqueness rule the ticket calls for, at the database layer rather than a race-prone
check-then-insert in application code. A revoked grant never counts against it: revoking first,
then granting again for the same membership and tool, succeeds and produces a second, unrelated
row -- `revoked_at` is a column on the row, never a delete.

`created_at` records when the grant was made; `revoked_at`/`revoked_by` are both nullable and
populated together, once, by revocation -- the row is never deleted, so a listing shows every
grant a tenant has ever made, active or not, alongside who granted and (if applicable) who revoked
it.

Grants for `app`: `SELECT, INSERT, UPDATE` -- `UPDATE` because revoking a grant (setting
`revoked_at`/`revoked_by` on an existing row) is the one mutation this table's own rules allow
after creation, exactly like `pending_actions` resolving from `pending` to a terminal status. No
`DELETE` grant, matching `pending_actions` and `agent_credentials`: the only way a row disappears
is the cascading delete from its tenant, never an application code path.
"""

from collections.abc import Sequence

from alembic import op

_TABLES_AT_THIS_REVISION: tuple[str, ...] = ("standing_grants",)

revision: str = "0031"
down_revision: str | None = "0022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE standing_grants (
            id                    uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id             uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            agent_membership_id   uuid NOT NULL REFERENCES memberships (id),
            tool_name             varchar(200) NOT NULL,
            granted_by            uuid NOT NULL REFERENCES memberships (id),
            created_at            timestamptz NOT NULL DEFAULT now(),
            revoked_at            timestamptz,
            revoked_by            uuid REFERENCES memberships (id)
        )
        """
    )
    op.execute("CREATE INDEX standing_grants_tenant_idx ON standing_grants (tenant_id)")
    op.execute(
        """
        CREATE UNIQUE INDEX standing_grants_active_uidx
            ON standing_grants (agent_membership_id, tool_name)
            WHERE revoked_at IS NULL
        """
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
                GRANT SELECT, INSERT, UPDATE ON standing_grants TO app;
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS standing_grants")
