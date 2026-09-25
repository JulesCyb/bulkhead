"""Constrain isolation tier and database alias in the control plane (Spec 10, #73).

Revision ID: 0005
Revises: 0002
Create Date: 2026-09-25

ADR-0002 (hybrid tenant isolation) decided that a tenant's isolation tier and, once it is
`dedicated`, the alias of the database it lives in are control-plane facts: written only by the
operator, read by the application through the existing `control.tenants_view`, and never
touched by a tenant's own request. Until now `control.tenants` (#12/0002) carried no columns for
either fact yet. This migration adds them and, per this ticket, ties them together with a
database-level constraint rather than trusting application code to keep them consistent:

- `isolation_tier` defaults every tenant to `'pooled'` (ADR-0002: pooled is the only tier any
  tenant has until an operator changes it) and is restricted to the two tiers CONTEXT.md names.
- `database_alias` is `NULL` for a pooled tenant and holds the alias string (never a DSN,
  hostname, or credential -- those live in a tenant secret file keyed by the alias, per
  ADR-0011) for a dedicated one.
- A `CHECK` constraint enforces "pooled implies null alias, dedicated implies non-null alias" at
  the schema level, so the guarantee holds even against a future bug in application code, not
  only against today's code path.

`control.tenants_view` (still `security_invoker`, still granted `SELECT` only to `app`, no new
grant added here) is extended to expose both new columns -- the same read-only surface Spec 10's
`tenant_session` routing (a later, separate ticket) will read from.

`control.database_aliases` is a small derived view enumerating the distinct aliases currently
referenced, substituting the reserved literal `'pooled'` for the `NULL` a pooled tenant carries,
so "which database aliases exist" is one query away for the migration runner (a later ticket)
without a dedicated aliases table -- not enough distinct facts exist yet to justify one.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE control.tenants
            ADD COLUMN isolation_tier text NOT NULL DEFAULT 'pooled',
            ADD COLUMN database_alias text
        """
    )
    op.execute(
        """
        ALTER TABLE control.tenants
            ADD CONSTRAINT control_tenants_isolation_tier_valid
                CHECK (isolation_tier IN ('pooled', 'dedicated'))
        """
    )
    op.execute(
        """
        ALTER TABLE control.tenants
            ADD CONSTRAINT control_tenants_alias_matches_tier
                CHECK (
                    (isolation_tier = 'pooled' AND database_alias IS NULL)
                    OR (isolation_tier = 'dedicated' AND database_alias IS NOT NULL)
                )
        """
    )

    # CREATE OR REPLACE VIEW: only appends columns, so the existing SELECT grant to `app`
    # (from 0002) keeps applying -- no new GRANT statement is needed or added.
    op.execute(
        """
        CREATE OR REPLACE VIEW control.tenants_view WITH (security_invoker = true) AS
            SELECT tenant_id, created_at, isolation_tier, database_alias FROM control.tenants
        """
    )

    # Not granted to `app`: enumerating aliases is an operator/migration-runner concern, never
    # something a tenant's own request needs.
    op.execute(
        """
        CREATE VIEW control.database_aliases WITH (security_invoker = true) AS
            SELECT DISTINCT COALESCE(database_alias, 'pooled') AS database_alias
            FROM control.tenants
        """
    )


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS control.database_aliases")
    op.execute(
        """
        CREATE OR REPLACE VIEW control.tenants_view WITH (security_invoker = true) AS
            SELECT tenant_id, created_at FROM control.tenants
        """
    )
    op.execute("ALTER TABLE control.tenants DROP CONSTRAINT control_tenants_alias_matches_tier")
    op.execute("ALTER TABLE control.tenants DROP CONSTRAINT control_tenants_isolation_tier_valid")
    op.execute(
        """
        ALTER TABLE control.tenants
            DROP COLUMN database_alias,
            DROP COLUMN isolation_tier
        """
    )
