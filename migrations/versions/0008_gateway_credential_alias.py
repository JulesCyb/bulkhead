"""Control-plane fact: a tenant's gateway-credential alias.

Revision ID: 0008
Revises: 0002
Create Date: 2026-09-25

Adds `gateway_credential_alias` to `control.tenants` (Spec 7 / #52, ADR-0009, ADR-0011): the
control plane records only the *alias* of a tenant's gateway credential, never the credential
itself, following the schema's own pattern from 0002 -- owned by `app_owner`, written only by
the owning role, exposed to `app` through the existing `security_invoker` view with `SELECT`
only. `app` gets no `INSERT`/`UPDATE`/`DELETE` on `control.tenants`; nothing here changes that.

`CREATE OR REPLACE VIEW` appending one column at the end keeps the already-granted `SELECT` on
`control.tenants_view` intact -- no new grant needed for `app` to see the added column.

Also fixes a latent bug in 0002's view, uncovered by this ticket's first real read through it as
`app`: `security_invoker = true` makes *every* permission check -- not only row security --
evaluate against the invoking role, so `app` would additionally need `SELECT` directly on
`control.tenants` itself (a grant this schema deliberately never gives), and every query through
the view failed with "permission denied for table tenants". `FORCE ROW LEVEL SECURITY` (set in
0002) already applies row security to `app_owner` too, and the policy's expression is a session
GUC (`current_setting('app.tenant_id', true)`), not anything role-dependent, so the plain,
security-definer view (the default -- `app_owner`'s own privileges, RLS still enforced by FORCE)
filters correctly for `app` without needing a direct table grant. `app`'s privileges stay exactly
`SELECT` on `control.tenants_view` and nothing else in `control`.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0008"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE control.tenants ADD COLUMN gateway_credential_alias text")
    op.execute(
        """
        CREATE OR REPLACE VIEW control.tenants_view WITH (security_invoker = false) AS
            SELECT tenant_id, created_at, gateway_credential_alias FROM control.tenants
        """
    )


def downgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE VIEW control.tenants_view WITH (security_invoker = true) AS
            SELECT tenant_id, created_at FROM control.tenants
        """
    )
    op.execute("ALTER TABLE control.tenants DROP COLUMN gateway_credential_alias")
