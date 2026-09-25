"""Tenant memberships replace per-tenant users (ADR-0003, Spec 2 / #23).

Revision ID: 0009
Revises: 0010
Create Date: 2026-09-25

The starter's `users` table bound a person to exactly one tenant (`UNIQUE (tenant_id, email)`)
and carried no relationship to the control-plane `control.identities` table added in 0003. This
migration drops `users` outright and replaces it with `memberships`: a tenant's roster of who
belongs to it, keyed by tenant plus a cross-schema reference to the global identity, carrying
that identity's role in the tenant.

`memberships` follows the same shape as every other `public` table in `TENANT_TABLES`: a
required, indexed `tenant_id NOT NULL REFERENCES tenants(id)`, `ENABLE`/`FORCE ROW LEVEL
SECURITY`, and the usual `tenant_id = current_setting('app.tenant_id', true)::uuid` policy
(USING and WITH CHECK). Two things are new relative to `users`:

- `identity_id` references `control.identities(id)` rather than carrying its own email/role
  identity data -- the membership is the tenant-scoped fact, the identity is the global one
  (ADR-0003). No RLS policy applies on the `control.identities` side of this reference; it never
  needed one (0003).
- `role` is restricted by a `CHECK` constraint to the four roles CONTEXT.md names: `admin`,
  `member`, `support` (an operator's membership for troubleshooting -- visible to every one of
  the tenant's admins through the same, unfiltered `SELECT` every other membership row gets, by
  construction: nothing here or in the repository layer filters by role), and `agent` (ADR-0005).
- `UNIQUE (tenant_id, identity_id)` replaces `users`' `UNIQUE (tenant_id, email)`: one membership
  per identity per tenant, matching the "a person in two tenants has two memberships" rule.

Grants for `app` mirror the retired `users` grants exactly: `SELECT, INSERT, UPDATE, DELETE`.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0009"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS users")

    op.execute(
        """
        CREATE TABLE memberships (
            id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id   uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            identity_id uuid NOT NULL REFERENCES control.identities(id),
            role        varchar(50) NOT NULL,
            created_at  timestamptz NOT NULL DEFAULT now(),
            UNIQUE (tenant_id, identity_id),
            CONSTRAINT memberships_role_valid
                CHECK (role IN ('admin', 'member', 'support', 'agent'))
        )
        """
    )
    op.execute("CREATE INDEX memberships_tenant_idx ON memberships (tenant_id)")

    op.execute("ALTER TABLE memberships ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE memberships FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY memberships_tenant_isolation ON memberships
            USING      (tenant_id = current_setting('app.tenant_id', true)::uuid)
            WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid)
        """
    )

    # Grants for the app role (exists only if 01-init.sh has run -- skip otherwise, matching
    # 0001's pattern for the table this one replaces).
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app') THEN
                GRANT SELECT, INSERT, UPDATE, DELETE ON memberships TO app;
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS memberships")

    op.execute(
        """
        CREATE TABLE users (
            id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id   uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            email       varchar(320) NOT NULL,
            role        varchar(50) NOT NULL DEFAULT 'member',
            created_at  timestamptz NOT NULL DEFAULT now(),
            UNIQUE (tenant_id, email)
        )
        """
    )
    op.execute("CREATE INDEX users_tenant_idx ON users (tenant_id)")
    op.execute("ALTER TABLE users ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE users FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY users_tenant_isolation ON users
            USING      (tenant_id = current_setting('app.tenant_id', true)::uuid)
            WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid)
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app') THEN
                GRANT SELECT, INSERT, UPDATE, DELETE ON users TO app;
            END IF;
        END $$;
        """
    )
