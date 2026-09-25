"""Base schema: tenants, users, documents (pgvector) with Row-Level Security.

Revision ID: 0001
Revises:
Create Date: 2026-08-17

Roles: migrations run as the owner (DATABASE_URL_MIGRATIONS). The app connects as the `app`
role (no superuser, NOBYPASSRLS — created in docker/postgres/01-init.sh). Superusers always
bypass RLS; that is why the app must never run as one.
"""

from collections.abc import Sequence

from alembic import op

from app.db.tenant_tables import TENANT_TABLES

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _rls(table: str) -> None:
    op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
    op.execute(
        f"""
        CREATE POLICY {table}_tenant_isolation ON {table}
            USING      (tenant_id = current_setting('app.tenant_id', true)::uuid)
            WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid)
        """
    )


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute(
        """
        CREATE TABLE tenants (
            id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            name        varchar(200) NOT NULL,
            settings    jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at  timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("ALTER TABLE tenants ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE tenants FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY tenants_self_only ON tenants
            USING (id = current_setting('app.tenant_id', true)::uuid)
        """
    )

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

    op.execute(
        """
        CREATE TABLE documents (
            id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id   uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            title       varchar(500) NOT NULL,
            content     text NOT NULL,
            embedding   vector(1536),
            metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at  timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX documents_tenant_idx ON documents (tenant_id)")
    # A vector index is only needed from a few tens of thousands of rows; then e.g.:
    # CREATE INDEX documents_embedding_idx ON documents USING hnsw (embedding vector_cosine_ops)

    for table in TENANT_TABLES:
        _rls(table)

    # Grants for the app role (exists only if 01-init.sh has run — skip otherwise).
    # tenants: no INSERT/DELETE for the app — creating tenants is an owner/admin operation,
    # and DELETE would cascade an entire tenant away in one statement. UPDATE stays so the
    # app can maintain tenants.settings.
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app') THEN
                GRANT SELECT, INSERT, UPDATE, DELETE ON users, documents TO app;
                GRANT SELECT, UPDATE ON tenants TO app;
                -- The default privileges from 01-init.sh grant full DML on every new table,
                -- so the narrower tenants grant must be enforced with an explicit REVOKE.
                REVOKE INSERT, DELETE ON tenants FROM app;
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS documents")
    op.execute("DROP TABLE IF EXISTS users")
    op.execute("DROP TABLE IF EXISTS tenants")
