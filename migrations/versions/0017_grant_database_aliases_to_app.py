"""Let `app` enumerate database aliases for the runtime guard (Spec 10, #77).

Revision ID: 0017
Revises: 0016
Create Date: 2026-09-25

The fail-closed runtime guard (ADR-0011, issue #15) runs against every database engine the
process might open -- the pooled one plus every dedicated alias the control plane references.
To learn those aliases, the guard (connected as `app`) needs one cross-tenant read of
`control.tenants`, which carries FORCE ROW LEVEL SECURITY.

That read goes through `control.enumerate_database_aliases()` (0016): a SECURITY DEFINER
function owned by `app_owner` that returns alias strings only, and whose policy is satisfied
only while `current_user = 'app_owner'` inside it. This migration grants `app` EXECUTE on it and
nothing else -- `app`'s view of `control.tenants_view` stays limited to its own tenant's row, and
to no row at all without a tenant context.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0017"
down_revision: str | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app') THEN
                GRANT EXECUTE ON FUNCTION control.enumerate_database_aliases() TO app;
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app') THEN
                REVOKE EXECUTE ON FUNCTION control.enumerate_database_aliases() FROM app;
            END IF;
        END $$;
        """
    )
