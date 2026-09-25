"""Control-plane fact: a tenant's residency (Spec 7 / #54, ADR-0008, ADR-0009).

Revision ID: 0013
Revises: 0008
Create Date: 2026-09-25

ADR-0008 decided residency selects a tenant's model, embedding, and tracing routes; CONTEXT.md's
own "Control plane" entry already names residency as one of the operator-owned facts -- "owned
by the operator, never by a tenant" -- alongside isolation tier and database location. Until now
`control.tenants` (0002/0005/0008) carried neither: nothing recorded a tenant's residency as a
control-plane fact the application could read, so #54's model-allow-list check (validating a
tenant's chosen model against the allow-list for *its* residency, `RESIDENCY_MODEL_ALLOW_LIST` in
`app/config.py`) had no tenant-scoped fact to check it against.

Follows the exact pattern 0005 set for `isolation_tier`: a `NOT NULL` column with a deployment
default (`'eu'`, matching `Settings.residency`'s own default) and a `CHECK` constraint
restricting it to the residencies this deployment's configuration actually knows about
(`RESIDENCY_ALLOW_LIST`'s keys) -- adding a residency is a migration here plus a config entry,
never a silent, unchecked string. `control.tenants_view` (still granted `SELECT`-only to `app`,
no new grant needed here, same as 0008's own column addition) is extended to expose it.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0013"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE control.tenants ADD COLUMN residency text NOT NULL DEFAULT 'eu'")
    op.execute(
        """
        ALTER TABLE control.tenants
            ADD CONSTRAINT control_tenants_residency_valid
                CHECK (residency IN ('eu', 'us'))
        """
    )
    op.execute(
        """
        CREATE OR REPLACE VIEW control.tenants_view WITH (security_invoker = false) AS
            SELECT tenant_id, created_at, isolation_tier, database_alias,
                   gateway_credential_alias, residency
            FROM control.tenants
        """
    )


def downgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE VIEW control.tenants_view WITH (security_invoker = false) AS
            SELECT tenant_id, created_at, isolation_tier, database_alias, gateway_credential_alias
            FROM control.tenants
        """
    )
    op.execute("ALTER TABLE control.tenants DROP CONSTRAINT control_tenants_residency_valid")
    op.execute("ALTER TABLE control.tenants DROP COLUMN residency")
