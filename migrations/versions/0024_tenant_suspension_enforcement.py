"""Suspension enforced at context resolution, and the operator's suspend/unsuspend write path.

Revision ID: 0024
Revises: 0032
Create Date: 2026-09-25

Spec 9 / #69, ADR-0010. Two changes:

1. `control.tenants_view` (0002/0008/0013) gains `suspended`/`suspended_at` (0004's generated
   flag and its source timestamp), so `app/db/session.py`'s `tenant_session()` -- the same
   tenant-scoped control-plane read every live request already makes to pick which database
   serves it -- can reject before ever opening a session against the tenant's own data. No new
   grant needed for `app`: the view is already `SELECT`-granted, and `FORCE ROW LEVEL SECURITY`
   on `control.tenants` still filters every row by `app.tenant_id`, so this exposes only the
   querying tenant's own suspension state, never another's.

2. The operator tool's `suspend`/`unsuspend` commands need to *write*
   `control.tenants.suspended_at`, which `app_owner`'s table ownership does not by itself allow:
   `FORCE ROW LEVEL SECURITY` (0002) binds the owner too, and the only existing policy on
   `control.tenants` (`control_tenants_tenant_isolation`) is scoped to a *tenant's own*
   `app.tenant_id`, which the operator tool never sets. This follows the exact escape-hatch shape
   0012 established for the operator's read-only listing (`current_user = 'app_owner'` plus a
   dedicated flag, checked with `current_user` rather than `session_user` for the same reason
   0012/0016's docstrings give: a `SECURITY DEFINER` function reassigns `current_user` to its
   owner, a plain view does not) -- but with its own flag (`app.control_operator_write`, kept
   separate from the read-only listing's `app.control_operator_read` so suspending a tenant can
   never be triggered merely by widening a read), and `FOR ALL` rather than `FOR SELECT`, since
   this one also has to write:

   - `control_tenants_operator_write` -- a policy on `control.tenants`, `FOR ALL`, gated on
     `current_user = 'app_owner' AND current_setting('app.control_operator_write', true) = 'true'`.
   - `control.set_tenant_suspended(tenant_id uuid, p_suspended boolean)` -- a `SECURITY DEFINER`
     function, owned by `app_owner`, that sets the flag `is_local=true`, upserts a `control.tenants`
     row for the tenant if none exists yet (a pooled tenant provisioned only through
     `scripts/seed.py`'s default path may have none -- ADR-0002's pooled default, not an error),
     reads the current `suspended_at` `FOR UPDATE`, writes the new value only if it actually
     changes (re-suspending an already-suspended tenant, or un-suspending an already-active one,
     changes nothing -- the caller inspects the returned `changed` column to report a no-op
     instead of "ok"), and restores the caller's previous flag value before returning --
     mirroring `control.enumerate_tenants()` (0012)'s own restore-after-read guarantee. Not
     granted to `app`: this write is operator-only, like the listing it complements.

3. `control.enumerate_tenants()` (0012) is replaced with a `LEFT JOIN` version. Its original
   `INNER JOIN` between `control.tenants` and `public.tenants` meant a pooled tenant with no
   `control.tenants` row at all -- ADR-0002's own default, and exactly the state `suspend` above
   is built to handle without erroring -- was invisible to the operator tool's `list` command and,
   through `app/operator/lookup.py`'s `resolve_tenant`, unresolvable by `suspend`/`unsuspend`
   either: an operator could not suspend the most common kind of tenant this template ships. The
   replacement reports every tenant in `public.tenants`, `COALESCE`-ing the same pooled default
   `app/db/session.py`'s own control-plane read already uses (`isolation_tier` -> `'pooled'`,
   `suspended` -> `false`) when no `control.tenants` row exists yet, rather than a second,
   silently different notion of "the default" living only in this function.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0024"
down_revision: str | None = "0032"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE VIEW control.tenants_view WITH (security_invoker = false) AS
            SELECT tenant_id, created_at, isolation_tier, database_alias,
                   gateway_credential_alias, residency, suspended, suspended_at
            FROM control.tenants
        """
    )

    op.execute(
        """
        CREATE POLICY control_tenants_operator_write ON control.tenants
            FOR ALL
            USING (
                current_user = 'app_owner'
                AND current_setting('app.control_operator_write', true) = 'true'
            )
            WITH CHECK (
                current_user = 'app_owner'
                AND current_setting('app.control_operator_write', true) = 'true'
            )
        """
    )

    op.execute(
        """
        CREATE FUNCTION control.set_tenant_suspended(p_tenant_id uuid, p_suspended boolean)
        RETURNS TABLE(changed boolean, suspended_at timestamptz)
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = control, public, pg_temp
        AS $$
        DECLARE
            caller_flag text := current_setting('app.control_operator_write', true);
            prev_suspended_at timestamptz;
            new_suspended_at timestamptz;
        BEGIN
            PERFORM set_config('app.control_operator_write', 'true', true);

            -- A tenant provisioned only through the pooled default (ADR-0002) may have no
            -- control.tenants row at all yet; suspending it is not an error, it creates one.
            INSERT INTO control.tenants (tenant_id) VALUES (p_tenant_id)
            ON CONFLICT (tenant_id) DO NOTHING;

            SELECT ct.suspended_at INTO prev_suspended_at
            FROM control.tenants ct
            WHERE ct.tenant_id = p_tenant_id
            FOR UPDATE;

            new_suspended_at := CASE
                WHEN p_suspended THEN COALESCE(prev_suspended_at, now())
                ELSE NULL
            END;

            UPDATE control.tenants ct SET suspended_at = new_suspended_at
            WHERE ct.tenant_id = p_tenant_id;

            -- Restore the caller's previous setting, exactly like control.enumerate_tenants().
            PERFORM set_config('app.control_operator_write', coalesce(caller_flag, ''), true);

            RETURN QUERY SELECT
                (prev_suspended_at IS DISTINCT FROM new_suspended_at),
                new_suspended_at;
        END;
        $$
        """
    )
    op.execute("REVOKE ALL ON FUNCTION control.set_tenant_suspended(uuid, boolean) FROM PUBLIC")

    op.execute(
        """
        CREATE OR REPLACE FUNCTION control.enumerate_tenants()
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
                    pt.id,
                    pt.name::text,
                    COALESCE(ct.isolation_tier, 'pooled'),
                    ct.database_alias,
                    ct.residency,
                    COALESCE(ct.suspended, false),
                    ct.suspended_at
                FROM public.tenants pt
                LEFT JOIN control.tenants ct ON ct.tenant_id = pt.id;
            PERFORM set_config('app.control_operator_read', coalesce(caller_flag, ''), true);
        END;
        $$
        """
    )


def downgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION control.enumerate_tenants()
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
            PERFORM set_config('app.control_operator_read', coalesce(caller_flag, ''), true);
        END;
        $$
        """
    )
    op.execute("DROP FUNCTION IF EXISTS control.set_tenant_suspended(uuid, boolean)")
    op.execute("DROP POLICY IF EXISTS control_tenants_operator_write ON control.tenants")
    # CREATE OR REPLACE cannot drop a view's trailing columns (`suspended`/`suspended_at`) --
    # only adding columns at the end is supported that way, same constraint 0002's own downgrade
    # docstring notes for ALTER POLICY. DROP + CREATE is the only way back to the narrower shape.
    op.execute("DROP VIEW control.tenants_view")
    op.execute(
        """
        CREATE VIEW control.tenants_view WITH (security_invoker = false) AS
            SELECT tenant_id, created_at, isolation_tier, database_alias,
                   gateway_credential_alias, residency
            FROM control.tenants
        """
    )
    # DROP VIEW above dropped the view's own grants along with it -- restore 0002's SELECT grant
    # to `app` (skipped, like 0002's own grant, when `app` doesn't exist -- test fixtures that
    # run migrations as the cluster superuser with no `app` role at all).
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app') THEN
                GRANT SELECT ON control.tenants_view TO app;
            END IF;
        END $$;
        """
    )
