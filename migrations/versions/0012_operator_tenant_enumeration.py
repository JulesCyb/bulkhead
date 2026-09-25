"""Operator-only cross-tenant enumeration for the operator tool's tenant listing (Spec 9 / #68).

Revision ID: 0012
Revises: 0013
Create Date: 2026-09-25

The operator tool's read-only listing needs to see *every* tenant's isolation tier, database
alias, and suspension state (`control.tenants`) alongside its residency
(`control.tenants.residency`, 0013, ADR-0008) -- as the owner role, without a
`RequestContext`/`app.tenant_id` to scope any one query. Both `control.tenants` and
`public.tenants` carry `FORCE ROW LEVEL SECURITY` (0002/0001), which -- unlike ordinary RLS --
binds `app_owner` too, so with no tenant context set, every ordinary query against either table
returns zero rows for the operator tool exactly as it would for anyone else.

0016 (#76) already solved the identical problem for the migration runner's alias enumeration.
Its first version gated the escape-hatch policy on `session_user = 'app_owner'`; it was then
tightened to `current_user = 'app_owner'` (see 0016's own current docstring/history) once testing
showed `current_user` is the check that actually distinguishes the two cases that matter here:
`SECURITY DEFINER` functions reassign `current_user` to the function's owner for the duration of
the call (so a function owned by `app_owner` satisfies the policy no matter who calls it), while a
plain, non-`security_invoker` view such as `control.tenants_view` (0008) does **not** reassign
`current_user` even though it checks table-level permissions as its owner -- so a real `app`
session querying that view still has `current_user = 'app'` and never satisfies this policy, flag
or no flag. `session_user` would have worked too (it never changes either way) but `current_user`
is what 0017 then relied on to safely grant `app` `EXECUTE` on the *function itself* without
widening the *view*: the same distinction this migration's function also depends on.

This migration reuses that exact shape rather than inventing a second one, with its own flag
(`app.control_operator_read`) so the operator tool's read is independent of the migration
runner's:

- `control_tenants_operator_read` -- a policy on `control.tenants`, `FOR SELECT`, gated on
  `current_user = 'app_owner' AND current_setting('app.control_operator_read', true) = 'true'`.
- `tenants_operator_read` -- the same shape on `public.tenants`, needed for the tenant's name,
  which lives there, not in `control.tenants`.
- `control.enumerate_tenants()` -- a `SECURITY DEFINER` function, owned by `app_owner`, that sets
  the flag `is_local=true`, reads every tenant's id, name, isolation tier, database alias,
  residency, and suspension state in one join across both tables, and restores the caller's
  previous flag value before returning -- mirroring `control.tenant_auth_settings` (0003) and
  `control.enumerate_database_aliases` (0016)'s own restore-after-read guarantee. Not granted to
  `app` (unlike 0017's grant of `enumerate_database_aliases`): this listing is operator/auditor
  only, nothing a tenant's own request needs, so there is no reason to widen it further. The
  operator tool's tenant-lookup helper (`app/operator/lookup.py`) also calls this function,
  filtering its result client-side by id or name, rather than adding a second enumeration
  mechanism just for lookup.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0012"
down_revision: str | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE POLICY control_tenants_operator_read ON control.tenants
            FOR SELECT
            USING (
                current_user = 'app_owner'
                AND current_setting('app.control_operator_read', true) = 'true'
            )
        """
    )
    op.execute(
        """
        CREATE POLICY tenants_operator_read ON public.tenants
            FOR SELECT
            USING (
                current_user = 'app_owner'
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
                    ct.residency,
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
