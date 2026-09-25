"""Tenant policies treat an empty `app.tenant_id` as "no context", not as an error.

Revision ID: 0040
Revises: 0038
Create Date: 2026-09-25

Once a pooled connection has carried a tenant context, Postgres keeps the custom setting
`app.tenant_id` defined for the rest of the session: after the transaction ends -- and after the
pool's RESET ALL -- `current_setting('app.tenant_id', true)` returns '' instead of NULL. Every
tenant policy casts that value with `::uuid`, so the first context-less read on a reused
connection (`control_session()`, the runtime guard behind `/ready`) failed with
`invalid input syntax for type uuid: ""` instead of seeing zero rows.

This migration rewrites every policy in `public` and `control` whose expression reads
`app.tenant_id` to use `NULLIF(current_setting('app.tenant_id', true), '')::uuid`. An empty
setting now means exactly what an unset one always meant: no row matches. New tenant tables use
the same expression (CLAUDE.md, rule 2).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0040"
down_revision: str | None = "0038"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# How Postgres deparses the old and new expressions in pg_policies.
_OLD = "(current_setting('app.tenant_id'::text, true))::uuid"
_NEW = "(NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid"


def _rewrite(old: str, new: str) -> None:
    op.execute(
        f"""
        DO $$
        DECLARE
            p record;
        BEGIN
            FOR p IN
                SELECT schemaname, tablename, policyname, qual, with_check
                FROM pg_policies
                WHERE schemaname IN ('public', 'control')
                  AND (position($old${old}$old$ IN coalesce(qual, '')) > 0
                       OR position($old${old}$old$ IN coalesce(with_check, '')) > 0)
            LOOP
                IF p.qual IS NOT NULL THEN
                    EXECUTE format(
                        'ALTER POLICY %I ON %I.%I USING (%s)',
                        p.policyname, p.schemaname, p.tablename,
                        replace(p.qual, $old${old}$old$, $new${new}$new$)
                    );
                END IF;
                IF p.with_check IS NOT NULL THEN
                    EXECUTE format(
                        'ALTER POLICY %I ON %I.%I WITH CHECK (%s)',
                        p.policyname, p.schemaname, p.tablename,
                        replace(p.with_check, $old${old}$old$, $new${new}$new$)
                    );
                END IF;
            END LOOP;
        END $$;
        """
    )


def upgrade() -> None:
    _rewrite(_OLD, _NEW)


def downgrade() -> None:
    _rewrite(_NEW, _OLD)
