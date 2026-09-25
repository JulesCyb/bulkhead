"""Tenant lifecycle: suspension on control.tenants, plus two append-only audit tables.

Revision ID: 0004
Revises: 0002
Create Date: 2026-09-25

Spec 9 / #66. `control.tenants` (0002) gains a suspension flag and timestamp -- the operator
tool's `suspend` step sets these, and later work (context resolution, #68) checks them on every
request; a suspended tenant loses access immediately while an erasure deadline runs (ADR-0010).

Two new tables live beside it, both owned by `app_owner` like the rest of `control`:

- `control.tenant_erasures` -- one row per completed erasure, written by the operator tool
  once it has removed a tenant's data everywhere ADR-0010 requires. It carries `tenant_id` as a
  plain column, deliberately with **no** foreign key to `public.tenants`: the row must document
  and outlive a tenant row the erasure itself deletes.
- `control.operator_actions` -- one row per invocation of the operator tool (`create`,
  `suspend`, `erase`), an audit trail independent of the erasure record. Also no foreign key,
  for the same reason: an `erase` action's own log entry must survive the tenant it names.

Both tables are append-only by grant, not by trigger, and the restriction applies to the owner
too: table ownership in Postgres grants privileges exactly as if by GRANT, and those are
revocable from the owner like any other role's. After the REVOKE/GRANT pair below, `app_owner`
(the role that runs this migration in every real deployment) holds INSERT only -- no SELECT,
UPDATE, or DELETE, so it cannot amend or remove a record it just wrote, only add another one.
Postgres has no ALTER-based way to review this later; only another migration re-granting the
privilege can. SELECT is not granted to anyone here: it is meant to be handed to a distinct
auditor role when one is provisioned (`GRANT SELECT ON control.<table> TO <auditor role>` after
`GRANT USAGE ON SCHEMA control TO <auditor role>`), never bundled with write access. `app` gets
nothing on either table -- these are operator/audit surfaces, never read through the request
path.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APPEND_ONLY_TABLES = ("tenant_erasures", "operator_actions")


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE control.tenants
            ADD COLUMN suspended    boolean NOT NULL DEFAULT false,
            ADD COLUMN suspended_at timestamptz
        """
    )

    op.execute(
        """
        CREATE TABLE control.tenant_erasures (
            id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            -- No REFERENCES: this row must outlive the tenant row it documents.
            tenant_id   uuid NOT NULL,
            erased_at   timestamptz NOT NULL DEFAULT now(),
            details     jsonb NOT NULL DEFAULT '{}'::jsonb
        )
        """
    )
    op.execute("CREATE INDEX tenant_erasures_tenant_idx ON control.tenant_erasures (tenant_id)")

    op.execute(
        """
        CREATE TABLE control.operator_actions (
            id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            -- No REFERENCES, same reason as above: an `erase` action's log entry must outlive
            -- the tenant it names.
            tenant_id     uuid NOT NULL,
            action        text NOT NULL,
            performed_at  timestamptz NOT NULL DEFAULT now(),
            details       jsonb NOT NULL DEFAULT '{}'::jsonb
        )
        """
    )
    op.execute("CREATE INDEX operator_actions_tenant_idx ON control.operator_actions (tenant_id)")

    for table in APPEND_ONLY_TABLES:
        # CURRENT_USER, not the literal "app_owner": whichever role runs this migration owns
        # the table it just created (migrations always run as app_owner outside tests -- see
        # DATABASE_URL_MIGRATIONS -- but some test fixtures run migrations as the cluster
        # superuser, which has no "app_owner" role to name and would bypass these grants
        # regardless). That role holds every privilege on the table implicitly, exactly as if
        # by GRANT ALL. REVOKE ALL strips that down to nothing, then GRANT INSERT hands back
        # only the one privilege the operator tool needs: it can add a record, never change or
        # remove one -- including its own.
        op.execute(f"REVOKE ALL ON control.{table} FROM CURRENT_USER")
        op.execute(f"GRANT INSERT ON control.{table} TO CURRENT_USER")
        op.execute(f"REVOKE ALL ON control.{table} FROM PUBLIC")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS control.operator_actions")
    op.execute("DROP TABLE IF EXISTS control.tenant_erasures")

    op.execute(
        """
        ALTER TABLE control.tenants
            DROP COLUMN suspended,
            DROP COLUMN suspended_at
        """
    )
