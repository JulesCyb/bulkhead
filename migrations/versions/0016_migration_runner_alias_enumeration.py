"""A migration-runner-only way to enumerate every database alias across all tenants.

Revision ID: 0016
Revises: 0009
Create Date: 2026-09-25

`control.database_aliases` (0005/#73) is a plain, `security_invoker` view over `control.tenants`,
which carries `FORCE ROW LEVEL SECURITY` (0002). That means the view is only ever readable
cross-tenant by a role that bypasses RLS outright (a cluster superuser) or a session that has
`app.tenant_id` set to every tenant in turn -- neither of which `scripts/migrate.py` (#76) can
use: `DATABASE_URL_MIGRATIONS` connects as `app_owner`, a real, `NOBYPASSRLS` role (ADR-0011), and
the whole point of enumerating aliases is reading *across* tenants in one query, not one at a
time. `control.tenant_auth_settings` (0003) solved the equivalent one-tenant problem by having a
`SECURITY DEFINER` function set `app.tenant_id` to the one row it needs, immediately before
reading, then restore the caller's previous value. Enumerating aliases needs to see every
tenant's row at once, so the same trick is applied with a dedicated escape-hatch policy instead
of the tenant-id one:

- A second, purely additive `SELECT` policy on `control.tenants`, permissive like the existing
  one (Postgres OR-combines permissive policies for the same command), gated on a
  transaction-local setting (`app.control_migration_read`) nothing else in this codebase ever
  sets. It grants no write access and changes no existing policy.
- `control.enumerate_database_aliases()`, a `SECURITY DEFINER` function owned by `app_owner`
  (exactly what runs migrations) that sets that one setting `is_local=true` immediately before
  reading every tenant's `database_alias` (`pooled` standing in for `NULL`, as the 0005 view
  already does) and restores the caller's previous value before returning -- so a later query in
  the same transaction never rides along on it, mirroring `tenant_auth_settings`'s own guarantee.

`control.database_aliases` (the view) is left exactly as 0005 built it: it still answers "which
aliases exist" correctly for a superuser or a per-tenant session, and nothing here changes its
grants. Not granted to `app`, same as 0005's view: enumerating aliases across every tenant is a
migration-runner concern, never something a tenant's own request needs.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0016"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE POLICY control_tenants_migration_read ON control.tenants
            FOR SELECT
            -- current_user, and only app_owner: any role may set a custom setting, and
            -- control.tenants_view runs with its owner's rights, so the flag alone would let
            -- `app` read every tenant's row through the view. A view does not change
            -- current_user; only a SECURITY DEFINER function owned by app_owner (such as
            -- enumerate_database_aliases below) or an app_owner login can satisfy this.
            USING (
                current_user = 'app_owner'
                AND current_setting('app.control_migration_read', true) = 'true'
            )
        """
    )

    op.execute(
        """
        CREATE FUNCTION control.enumerate_database_aliases()
        RETURNS TABLE(database_alias text)
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = control, pg_temp
        AS $$
        DECLARE
            caller_flag text := current_setting('app.control_migration_read', true);
        BEGIN
            PERFORM set_config('app.control_migration_read', 'true', true);
            RETURN QUERY
                SELECT DISTINCT COALESCE(t.database_alias, 'pooled') FROM control.tenants t;
            -- Restore the caller's previous setting: without this, a later query in the same
            -- transaction would keep seeing every tenant regardless of app.tenant_id.
            PERFORM set_config('app.control_migration_read', coalesce(caller_flag, ''), true);
        END;
        $$
        """
    )
    op.execute("REVOKE ALL ON FUNCTION control.enumerate_database_aliases() FROM PUBLIC")


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS control.enumerate_database_aliases()")
    op.execute("DROP POLICY IF EXISTS control_tenants_migration_read ON control.tenants")
