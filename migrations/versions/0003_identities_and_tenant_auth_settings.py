"""Global identities and per-tenant auth settings in the control plane.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-25

Adds `control.identities` (issuer, subject) -- ADR-0003's concrete instance of the control-plane
exception to CLAUDE.md rule 2 ("every table has tenant_id"): no `tenant_id` column, so no RLS
policy applies here at all. `app` reaches it only through `control.identity_lookup`, a
`SELECT`-only view exposing `id`, `issuer`, `subject` -- never `display_name`/`email`, and never
`INSERT`/`UPDATE`/`DELETE` against either the view or the underlying table.

`control.tenants` (Spec 1 / migration 0002) gains two columns: `identity_issuer` (nullable text;
the per-tenant token issuer -- NULL means "use the process-wide default", the interim
one-operator-run-provider case ADR-0003 keeps open) and `suspended_at` (nullable timestamptz).
Neither column is granted to `app` for writing -- only `app_owner` can set them.

`control.tenants` carries FORCE ROW LEVEL SECURITY (0002), which applies its tenant_id policy
even to `app_owner` (a non-superuser, NOBYPASSRLS role). A control-plane session deliberately
sets no tenant context (app/db/session.py's `control_session()`), so a SECURITY DEFINER function
owned by `app_owner` would still see zero rows for that same reason. `control.tenant_auth_settings`
works around this the one way that stays narrow: it sets `app.tenant_id` to its own
`p_tenant_id` argument, transaction-local only (`is_local=true`), immediately before reading --
satisfying the policy for exactly the one named tenant the caller asked about -- and restores the
caller's previous value before returning, so a later query in the same transaction never rides
along on it.
This function, plus `control.identity_lookup`, are the only two narrow reads a control-plane
session may make (see `control_session()`'s docstring); neither is ever used against a tenant's
own tables.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE control.identities (
            id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            issuer       text NOT NULL,
            subject      text NOT NULL,
            display_name text,
            email        text,
            created_at   timestamptz NOT NULL DEFAULT now(),
            UNIQUE (issuer, subject)
        )
        """
    )
    # No tenant_id, no RLS: this table is global by design (ADR-0003). The narrow grant below
    # is what keeps `app`'s access to it read-only and column-limited, not a policy.

    # No security_invoker here (unlike control.tenants_view): control.identities carries no
    # RLS policy to evaluate under the invoker's own context, and security_invoker would force
    # the underlying-table permission check onto the querying role too -- exactly what this
    # view exists to avoid. A plain view checks permissions as its owner (app_owner), so
    # granting SELECT on the view alone is enough; app never needs, and never gets, a grant on
    # control.identities itself.
    op.execute(
        """
        CREATE VIEW control.identity_lookup AS
            SELECT id, issuer, subject FROM control.identities
        """
    )

    op.execute("ALTER TABLE control.tenants ADD COLUMN identity_issuer text")
    op.execute("ALTER TABLE control.tenants ADD COLUMN suspended_at timestamptz")

    op.execute(
        """
        CREATE FUNCTION control.tenant_auth_settings(p_tenant_id uuid)
        RETURNS TABLE(identity_issuer text, suspended_at timestamptz)
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = control, pg_temp
        AS $$
        DECLARE
            caller_tenant text := current_setting('app.tenant_id', true);
        BEGIN
            PERFORM set_config('app.tenant_id', p_tenant_id::text, true);
            RETURN QUERY
                SELECT t.identity_issuer, t.suspended_at
                FROM control.tenants t
                WHERE t.tenant_id = p_tenant_id;
            -- Restore the caller's tenant context: without this, calling the function for
            -- tenant B inside tenant A's session would leave the rest of that transaction
            -- running as tenant B.
            PERFORM set_config('app.tenant_id', coalesce(caller_tenant, ''), true);
        END;
        $$
        """
    )
    op.execute("REVOKE ALL ON FUNCTION control.tenant_auth_settings(uuid) FROM PUBLIC")

    # Grants for the app role (exists only if 01-init.sh has run -- skip otherwise, matching
    # 0002's pattern). SELECT on the identity view, EXECUTE on the narrow settings function --
    # nothing else, on either object, ever.
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app') THEN
                GRANT SELECT ON control.identity_lookup TO app;
                GRANT EXECUTE ON FUNCTION control.tenant_auth_settings(uuid) TO app;
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
                REVOKE EXECUTE ON FUNCTION control.tenant_auth_settings(uuid) FROM app;
                REVOKE SELECT ON control.identity_lookup FROM app;
            END IF;
        END $$;
        """
    )
    op.execute("DROP FUNCTION IF EXISTS control.tenant_auth_settings(uuid)")
    op.execute("ALTER TABLE control.tenants DROP COLUMN IF EXISTS suspended_at")
    op.execute("ALTER TABLE control.tenants DROP COLUMN IF EXISTS identity_issuer")
    op.execute("DROP VIEW IF EXISTS control.identity_lookup")
    op.execute("DROP TABLE IF EXISTS control.identities")
