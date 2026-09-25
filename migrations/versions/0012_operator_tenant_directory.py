"""Operator-only tenant directory and lifecycle-info lookup (Spec 9 / #68).

Revision ID: 0012
Revises: 0008
Create Date: 2026-09-25

The operator tool's read-only tenant listing (#68) needs to enumerate *every* tenant and read
each one's isolation tier, residency, database alias, and suspension state -- as the owner role,
without a direct ad-hoc query. That is harder than it sounds: `control.tenants` carries `FORCE
ROW LEVEL SECURITY` (0002), which -- unlike ordinary RLS -- applies its `tenant_id =
current_setting('app.tenant_id', true)::uuid` policy to `app_owner` too, not only to `app`.
With no tenant context set (the operator tool takes no `RequestContext`), that policy matches
zero rows for every table it protects, including `public.tenants` itself, where a tenant's
residency actually lives (`tenants.settings->>'residency'`, ADR-0008). Reading "every tenant"
therefore first needs a way to *enumerate tenant ids* that isn't itself blocked by the same
policy, and then a way to read one known tenant's facts across both tables.

**Why not a role-scoped bypass policy on `control.tenants` itself** (`CREATE POLICY ... FOR
SELECT TO app_owner USING (true)`), the seemingly obvious fix? Because `control.tenants_view`
(0002, flipped to `security_invoker = false` in 0008 to fix a permission error for `app`) is a
plain view owned by `app_owner`: Postgres evaluates both the underlying table's privilege checks
*and* its row-security policies as the view's owner, not as the querying role, whenever
`security_invoker` is off. A policy scoped `TO app_owner` would therefore match every query `app`
runs through that view too -- not because `app` gained a grant, but because the view's own
row-security check silently runs as `app_owner` regardless of who is really connecting. That
would hand every tenant's row to any ordinary request through the exact view ADR-0011 built to
keep `app` to one tenant at a time. Never do this.

**The mechanism actually added here** is deliberately narrow and touches neither
`control.tenants` nor its existing policies or view:

- `control.tenant_directory` -- a new, tiny table with **no RLS policy at all** (like
  `control.identities`, ADR-0003's other global, unscoped control-plane table): `tenant_id`
  (FK to `public.tenants`, cascades on erasure) and `name`. It holds nothing sensitive -- a
  tenant's id and name, never isolation tier, residency, or anything else -- and is never
  granted to `app`, not even through a view; only `app_owner` (its owner) can read it. It is the
  one place the operator tool enumerates *which* tenants exist and resolves a name to an id,
  including for the tenant-lookup helper (#68) that must reject an ambiguous name: neither
  `public.tenants.name` nor this table has a uniqueness constraint on `name`, so two tenants can
  share a name and the lookup helper's ambiguity check is what actually protects against picking
  the wrong one. A later ticket populating `create` maintains this table alongside
  `control.tenants`; this migration only creates it.
- `control.tenant_lifecycle_info(p_tenant_id uuid)` -- a `SECURITY DEFINER` function following
  the exact restore-after-read pattern `control.tenant_auth_settings` (0003) already established
  for exactly this reason: it transaction-locally sets `app.tenant_id` to the *one* id the caller
  named, reads that tenant's row from both `control.tenants` and `public.tenants` (satisfying
  each one's own policy for exactly that tenant), and restores the caller's previous setting
  before returning. Called once per id from `control.tenant_directory`, this reconstructs the
  full listing one narrow, single-tenant read at a time -- never a cross-tenant policy bypass.
  Not granted to `app`; only `app_owner` may call it.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0012"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE control.tenant_directory (
            tenant_id   uuid PRIMARY KEY REFERENCES public.tenants(id) ON DELETE CASCADE,
            name        text NOT NULL,
            created_at  timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    # Deliberately no RLS here at all (see module docstring) -- and no grant to `app`, ever.
    op.execute("CREATE INDEX tenant_directory_name_idx ON control.tenant_directory (name)")

    op.execute(
        """
        CREATE FUNCTION control.tenant_lifecycle_info(p_tenant_id uuid)
        RETURNS TABLE(
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
            caller_tenant text := current_setting('app.tenant_id', true);
        BEGIN
            PERFORM set_config('app.tenant_id', p_tenant_id::text, true);
            RETURN QUERY
                SELECT
                    pt.name::text,
                    ct.isolation_tier,
                    ct.database_alias,
                    pt.settings ->> 'residency',
                    ct.suspended,
                    ct.suspended_at
                FROM control.tenants ct
                JOIN public.tenants pt ON pt.id = ct.tenant_id
                WHERE ct.tenant_id = p_tenant_id;
            -- Restore the caller's own context, exactly like control.tenant_auth_settings
            -- (0003): otherwise the rest of this transaction would keep running as
            -- p_tenant_id after this function returns.
            PERFORM set_config('app.tenant_id', coalesce(caller_tenant, ''), true);
        END;
        $$
        """
    )
    op.execute("REVOKE ALL ON FUNCTION control.tenant_lifecycle_info(uuid) FROM PUBLIC")
    # No GRANT EXECUTE to `app` -- this is an operator-only read; `app_owner` (the function's
    # owner) may always call its own function without a separate grant.


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS control.tenant_lifecycle_info(uuid)")
    op.execute("DROP TABLE IF EXISTS control.tenant_directory")
