"""Agent identity creation (ADR-0005, Spec 6 / #46).

Revision ID: 0032
Revises: 0022
Create Date: 2026-09-25

`control.identities` gains a `kind` column (`'person'` or `'agent'`, default `'person'` so every
existing row -- created only through the seed/admin path ADR-0003 describes -- keeps meaning what
it always meant). This is the concrete field ADR-0005 calls "an identity of kind agent": nothing
before this migration distinguished a person's identity from an agent's at the row level.

`app` still has no `INSERT` grant on `control.identities` (issue #22, unchanged by this
migration) -- and none is added here. Instead, `control.create_agent_identity(p_display_name)` is
a narrow `SECURITY DEFINER` function, owned by `app_owner` (the table owner), that does exactly
two things inside the caller's already-open transaction:

1. Insert one `control.identities` row with `kind = 'agent'`. The issuer/subject pair is
   synthesized here (`'agent'` / a fresh random UUID) and never taken from a parameter, so nothing
   this function does can impersonate or collide with a real person's `(issuer, subject)`.
2. Insert that identity's `agent`-role membership into `public.memberships`, in the tenant the
   caller's own session is *already* scoped to -- read from `current_setting('app.tenant_id',
   true)`, the exact GUC `tenant_session(ctx)` (app/db/session.py) sets before this function ever
   runs, never a parameter the caller could set to a different tenant. `memberships` carries
   `FORCE ROW LEVEL SECURITY` (migration 0009), which still applies to `app_owner` (a
   non-superuser, `NOBYPASSRLS` role) even though the function is `SECURITY DEFINER`; the INSERT
   satisfies that policy's `WITH CHECK` the same way every other tenant-scoped write in this
   codebase does -- by running inside a transaction where `app.tenant_id` is already set correctly
   -- rather than through any special-case trust of this function.

No `tenant_id` argument, no `issuer`/`subject` argument: the function's own signature is the
"only inserts agent identities, only in the caller's own tenant" guarantee, not a runtime check
this migration would otherwise have to get right in application code.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0032"
down_revision: str | None = "0022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE control.identities ADD COLUMN kind text NOT NULL DEFAULT 'person'")
    op.execute(
        """
        ALTER TABLE control.identities
            ADD CONSTRAINT identities_kind_valid CHECK (kind IN ('person', 'agent'))
        """
    )

    op.execute(
        """
        CREATE FUNCTION control.create_agent_identity(p_display_name text)
        RETURNS TABLE(identity_id uuid, membership_id uuid)
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = control, public, pg_temp
        AS $$
        DECLARE
            v_tenant_id     uuid := nullif(current_setting('app.tenant_id', true), '')::uuid;
            v_identity_id   uuid;
            v_membership_id uuid;
        BEGIN
            IF v_tenant_id IS NULL THEN
                RAISE EXCEPTION 'create_agent_identity: no tenant context set';
            END IF;

            INSERT INTO control.identities (issuer, subject, display_name, kind)
            VALUES ('agent', gen_random_uuid()::text, p_display_name, 'agent')
            RETURNING id INTO v_identity_id;

            INSERT INTO public.memberships (tenant_id, identity_id, role)
            VALUES (v_tenant_id, v_identity_id, 'agent')
            RETURNING id INTO v_membership_id;

            RETURN QUERY SELECT v_identity_id, v_membership_id;
        END;
        $$
        """
    )
    op.execute("REVOKE ALL ON FUNCTION control.create_agent_identity(text) FROM PUBLIC")

    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app') THEN
                GRANT EXECUTE ON FUNCTION control.create_agent_identity(text) TO app;
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
                REVOKE EXECUTE ON FUNCTION control.create_agent_identity(text) FROM app;
            END IF;
        END $$;
        """
    )
    op.execute("DROP FUNCTION IF EXISTS control.create_agent_identity(text)")
    op.execute("ALTER TABLE control.identities DROP CONSTRAINT IF EXISTS identities_kind_valid")
    op.execute("ALTER TABLE control.identities DROP COLUMN IF EXISTS kind")
