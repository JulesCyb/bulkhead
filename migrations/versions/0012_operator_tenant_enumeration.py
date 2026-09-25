"""Operator-only cross-tenant enumeration for the operator tool's tenant listing (Spec 9 / #68).

Revision ID: 0012
Revises: 0016
Create Date: 2026-09-25

The operator tool's read-only listing needs to see *every* tenant's isolation tier, database
alias, and suspension state (`control.tenants`) alongside its residency
(`tenants.settings->>'residency'`, ADR-0008) -- as the owner role, without a
`RequestContext`/`app.tenant_id` to scope any one query. Both `control.tenants` and
`public.tenants` carry `FORCE ROW LEVEL SECURITY` (0002/0001), which -- unlike ordinary RLS --
binds `app_owner` too, so with no tenant context set, every ordinary query against either table
returns zero rows for the operator tool exactly as it would for anyone else.

0016 (#76) already solved the identical problem for the migration runner's alias enumeration,
and after a real regression there (see its own docstring and
`test_app_cannot_widen_its_control_plane_view_with_the_migration_read_flag` in
`tests/test_rls_integration.py`), settled on the safe shape: a second, purely additive `SELECT`
policy gated on **both** a transaction-local flag *and* `session_user = 'app_owner'`. The
`session_user` check is load-bearing, not decorative: `control.tenants_view` (0008) runs with its
owner's (`app_owner`'s) privileges for both permission and row-security checks, and any role --
including `app` -- can set any custom setting in its own session. A flag-only policy would
therefore let `app` read every tenant's row through the existing view just by setting the flag
itself; gating on `session_user = 'app_owner'` closes that, because `session_user` reflects the
actual login role and is never affected by a view or a `SECURITY DEFINER` function's rights.

This migration reuses that exact shape rather than inventing a second one, with its own flag
(`app.control_operator_read`) so the operator tool's read is independent of the migration
runner's:

- `control_tenants_operator_read` -- a policy on `control.tenants`, `FOR SELECT`, gated on
  `session_user = 'app_owner' AND current_setting('app.control_operator_read', true) = 'true'`.
- `tenants_operator_read` -- the same shape on `public.tenants`, needed because residency lives
  there, not in `control.tenants`.
- `control.enumerate_tenants()` -- a `SECURITY DEFINER` function, owned by `app_owner`, that sets
  the flag `is_local=true`, reads every tenant's id, name, isolation tier, database alias,
  residency, and suspension state in one join across both tables, and restores the caller's
  previous flag value before returning -- mirroring `control.tenant_auth_settings` (0003) and
  `control.enumerate_database_aliases` (0016)'s own restore-after-read guarantee. Not granted to
  `app`; only `app_owner` (the function's owner) may call it. The operator tool's tenant-lookup
  helper (`app/operator/lookup.py`) also calls this function, filtering its result client-side by
  id or name, rather than adding a second enumeration mechanism just for lookup.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0012"
down_revision: str | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE POLICY control_tenants_operator_read ON control.tenants
            FOR SELECT
            USING (
                session_user = 'app_owner'
                AND current_setting('app.control_operator_read', true) = 'true'
            )
        """
    )
    op.execute(
        """
        CREATE POLICY tenants_operator_read ON public.tenants
            FOR SELECT
            USING (
                session_user = 'app_owner'
                AND current_setting('app.control_operator_read', true) = 'true'
            )
        """
    )

    op.execute(
        """
        CREATE FUNCTION control.enumerate_tenants()
        RETURNS TABLE(
            tenant_id       uuid,
            name            text,
            isolation_tier  text,
            database_alias  text,
            residency       text,
            suspended       boolean,
            suspended_at    timestamptz
        )
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = control, public, pg_temp
        AS $$
        DECLARE
            caller_flag text := current_setting('app.control_operator_read', true);
        BEGIN
            PERFORM set_config('app.control_operator_read', 'true', true);
            RETURN QUERY
                SELECT
                    ct.tenant_id,
                    pt.name::text,
                    ct.isolation_tier,
                    ct.database_alias,
                    pt.settings ->> 'residency',
                    ct.suspended,
                    ct.suspended_at
                FROM control.tenants ct
                JOIN public.tenants pt ON pt.id = ct.tenant_id;
            -- Restore the caller's previous setting: without this, a later query in the same
            -- transaction would keep seeing every tenant regardless of app.tenant_id.
            PERFORM set_config('app.control_operator_read', coalesce(caller_flag, ''), true);
        END;
        $$
        """
    )
    op.execute("REVOKE ALL ON FUNCTION control.enumerate_tenants() FROM PUBLIC")


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS control.enumerate_tenants()")
    op.execute("DROP POLICY IF EXISTS tenants_operator_read ON public.tenants")
    op.execute("DROP POLICY IF EXISTS control_tenants_operator_read ON control.tenants")
