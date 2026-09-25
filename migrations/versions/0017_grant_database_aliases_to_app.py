"""Let `app` enumerate database aliases through control.database_aliases (Spec 10, #77).

Revision ID: 0017
Revises: 0008
Create Date: 2026-09-25

The fail-closed runtime guard (ADR-0011, issue #15) is extended by this ticket to run against
every database engine the process might open -- the pooled one always, plus every dedicated
alias the control plane currently references -- not only the one `DATABASE_URL` it checked
before. To find out which aliases are currently referenced, the guard (running as the `app`
role, the only role the application ever connects as, over a `control_session()` -- no tenant
context set) needs to read `control.database_aliases` (migration 0005 / #73). That view was
deliberately not granted to `app` when it was created, because until now enumerating aliases was
only an operator/migration-runner concern queried with pgserver's own bootstrap superuser in
tests.

Two changes are needed to actually make that read work, not just to grant it:

1. **`security_invoker = false`.** Exactly the same latent bug 0008 already found and fixed for
   `control.tenants_view`: a `security_invoker = true` view checks the *invoking* role's
   privileges against the underlying base relation too, so `app` would additionally need
   `SELECT` directly on `control.tenants` -- a grant this schema deliberately never gives.
   Recreating `control.database_aliases` as a plain (security-definer) view means the query runs
   with `app_owner`'s own privileges instead, exactly like `control.tenants_view` since 0008.

2. **A second, narrow `SELECT` policy.** `control.tenants` carries `FORCE ROW LEVEL SECURITY`
   (0002), which applies its `tenant_id = current_setting('app.tenant_id', true)::uuid` policy
   even to `app_owner` -- there is no way for a non-superuser, `NOBYPASSRLS` role (by design,
   ADR-0011) to see rows outside that filter, `SECURITY DEFINER` or not, once `FORCE` is set.
   `control.tenant_auth_settings` (0003) works around this for a *single, named* tenant by
   setting `app.tenant_id` to that one argument before reading. Enumerating aliases needs the
   opposite: visibility across every tenant, with no single id to scope to. `control_session()`
   (`app/db/session.py`) is exactly the session mode reserved for this shape of read -- no
   tenant context is ever set inside it, by construction, for exactly the narrow, cross-tenant
   control-plane reads `app` is granted (today: `control.identity_lookup`,
   `control.tenant_auth_settings`; from this migration on: `control.database_aliases`). The new
   policy grants `SELECT` visibility across every row precisely when
   `current_setting('app.tenant_id', true) IS NULL` -- i.e., precisely inside a
   `control_session()`, never inside a `tenant_session(ctx)` (which always sets a tenant id
   first). A stray raw query issued with no session wrapper at all would also see every row
   under this policy; nothing in the codebase does that (every control-plane read goes through
   `tenant_session` or `control_session`), and this is no looser than the same class of "a
   session mode exists so its narrow reads work at all" trade-off `control_session()` already
   makes for the two reads it already permits.

No column beyond `database_alias` is exposed by `control.database_aliases`, and no grant is
added on the underlying `control.tenants` table itself -- only on the view, and only `SELECT`.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0017"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE POLICY control_tenants_no_context_enumeration ON control.tenants
            FOR SELECT
            USING (current_setting('app.tenant_id', true) IS NULL)
        """
    )
    op.execute(
        """
        CREATE OR REPLACE VIEW control.database_aliases WITH (security_invoker = false) AS
            SELECT DISTINCT COALESCE(database_alias, 'pooled') AS database_alias
            FROM control.tenants
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app') THEN
                GRANT SELECT ON control.database_aliases TO app;
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
                REVOKE SELECT ON control.database_aliases FROM app;
            END IF;
        END $$;
        """
    )
    op.execute(
        """
        CREATE OR REPLACE VIEW control.database_aliases WITH (security_invoker = true) AS
            SELECT DISTINCT COALESCE(database_alias, 'pooled') AS database_alias
            FROM control.tenants
        """
    )
    op.execute("DROP POLICY control_tenants_no_context_enumeration ON control.tenants")
