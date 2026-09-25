"""Control-plane schema: an operator-only home for facts about a tenant a tenant's own
request must never change.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-25

A new `control` schema, owned by `app_owner` (never `app`), holds `control.tenants` — keyed by
`tenant_id`, carrying no feature columns yet beyond `tenant_id`/`created_at`. Later specs
(residency, isolation tier, database location, suspension state) add their own columns here,
following this same pattern. `app` reaches it only through a `security_invoker` view granted
`SELECT`; it gets no `INSERT`/`UPDATE`/`DELETE` on anything in `control`, ever.

`public.tenants` gets narrower too: the app's role-wide `UPDATE` grant is replaced by a
column-level grant on `settings` alone -- the one field a tenant may legitimately change on
itself. Any future column added to `tenants` is unwritable by `app` by default. The existing
`tenants_self_only` policy gains a `WITH CHECK` clause matching its `USING` clause, so a write
targeting a mismatched tenant is rejected by the policy itself, symmetric with how reads are
already restricted.

Creating a new schema needs `CREATE` on the database itself (schema *ownership* alone, as
`public` already has, is not enough) -- see the matching grant in `docker/postgres/01-init.sh`.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA control")

    op.execute(
        """
        CREATE TABLE control.tenants (
            tenant_id   uuid PRIMARY KEY REFERENCES public.tenants(id) ON DELETE CASCADE,
            created_at  timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("ALTER TABLE control.tenants ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE control.tenants FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY control_tenants_tenant_isolation ON control.tenants
            USING      (tenant_id = current_setting('app.tenant_id', true)::uuid)
            WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid)
        """
    )

    # security_invoker: the querying role's own RLS context applies, not the view owner's
    # (app_owner's, which would otherwise see every tenant regardless of app.tenant_id).
    op.execute(
        """
        CREATE VIEW control.tenants_view WITH (security_invoker = true) AS
            SELECT tenant_id, created_at FROM control.tenants
        """
    )

    # Grants for the app role (exists only if 01-init.sh has run -- skip otherwise). SELECT on
    # the view only: no INSERT/UPDATE/DELETE on anything in `control`, ever.
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app') THEN
                GRANT USAGE ON SCHEMA control TO app;
                GRANT SELECT ON control.tenants_view TO app;

                -- public.tenants: the app role's table-wide UPDATE grant (from 0001) is
                -- replaced by a column-level one on `settings` alone -- the one column a
                -- tenant may legitimately change on itself. Any future column added to
                -- `tenants` is unwritable by `app` until explicitly granted.
                REVOKE UPDATE ON public.tenants FROM app;
                GRANT UPDATE (settings) ON public.tenants TO app;
            END IF;
        END $$;
        """
    )

    # tenants_self_only (from 0001) only restricted reads; writes now validated the same way.
    op.execute(
        """
        ALTER POLICY tenants_self_only ON public.tenants
            USING      (id = current_setting('app.tenant_id', true)::uuid)
            WITH CHECK (id = current_setting('app.tenant_id', true)::uuid)
        """
    )


def downgrade() -> None:
    # ALTER POLICY has no way to drop a WITH CHECK clause once added; recreate the policy in
    # its 0001 shape (USING only) instead.
    op.execute("DROP POLICY tenants_self_only ON public.tenants")
    op.execute(
        """
        CREATE POLICY tenants_self_only ON public.tenants
            USING (id = current_setting('app.tenant_id', true)::uuid)
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app') THEN
                REVOKE UPDATE (settings) ON public.tenants FROM app;
                GRANT UPDATE ON public.tenants TO app;
                REVOKE SELECT ON control.tenants_view FROM app;
                REVOKE USAGE ON SCHEMA control FROM app;
            END IF;
        END $$;
        """
    )
    op.execute("DROP VIEW IF EXISTS control.tenants_view")
    op.execute("DROP TABLE IF EXISTS control.tenants")
    op.execute("DROP SCHEMA IF EXISTS control")
