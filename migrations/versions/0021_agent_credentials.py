"""Agent identity credentials (ADR-0005, Spec 6 / #45).

Revision ID: 0021
Revises: 0012
Create Date: 2026-09-25

An agent identity's issued credential lives in its own tenant-scoped, durable record --
`agent_credentials`. It follows the standard tenant-table shape (CLAUDE.md rule 2, same as
`memberships` in 0009): a required, indexed `tenant_id NOT NULL REFERENCES tenants(id)`,
`ENABLE`/`FORCE ROW LEVEL SECURITY`, and the usual `tenant_id = current_setting('app.tenant_id',
true)::uuid` policy (USING and WITH CHECK).

Two columns carry the credential itself, kept deliberately distinct (issue #45, user story 34):

- `public_id`: the identifier a presented credential carries in the open. `UNIQUE (tenant_id,
  public_id)` makes verification a direct, indexed lookup by identifier rather than a scan over
  every credential the tenant has issued.
- `secret_hash`: SHA-256 of the actual secret, hex-encoded, and nothing else -- the plaintext
  secret itself is never persisted anywhere, at any point after creation (see
  `app/repositories/agent_credentials.py` for why SHA-256 rather than a slow, salted KDF like
  argon2/scrypt is the right hash here: the input is a 256-bit CSPRNG-generated token, not a
  human-chosen password).

`identity_id` references `control.identities(id)` -- the agent identity (ADR-0005) this
credential authenticates as -- exactly the same cross-schema FK pattern `memberships.identity_id`
already uses, resolved against the referenced table owner's privileges (see 0010's docstring),
needing no grant on `control.identities` for `app`.

`created_by` follows the audit-column convention 0010 established: a `NOT NULL` FK to
`control.identities`, defaulted from the per-transaction `app.identity_id` setting
`tenant_session()` sets -- the *admin* identity that issued the credential, distinct from
`identity_id` (the *agent* identity the credential belongs to). Unlike `documents`, there is no
`updated_by`/update-audit trigger here: the only mutation this table's own rules allow after
creation is revocation (`revoked_at`) and touching `last_used_at` on a successful verification,
neither of which is "who last edited this row" in the sense 0010 was solving for.

`revoked_at` and `last_used_at` are both nullable timestamps: revoking sets the former without
ever deleting the row (issue #45's "revoking marks a credential dead without deleting its
history"), and a successful verification advances the latter -- a failed one must not. No
`DELETE` grant is given to `app` at all (unlike `memberships`/`documents`, which do get one): the
only way this table's history disappears is tenant erasure (Spec 9), never an application code
path, so the grant list itself enforces the "never deleted" half of the invariant rather than
relying on `app/repositories/agent_credentials.py` alone to honor it.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0021"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE agent_credentials (
            id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id    uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            identity_id  uuid NOT NULL REFERENCES control.identities (id),
            name         varchar(200) NOT NULL,
            public_id    varchar(64) NOT NULL,
            secret_hash  varchar(128) NOT NULL,
            created_at   timestamptz NOT NULL DEFAULT now(),
            created_by   uuid NOT NULL
                REFERENCES control.identities (id)
                DEFAULT current_setting('app.identity_id', true)::uuid,
            revoked_at   timestamptz,
            last_used_at timestamptz,
            UNIQUE (tenant_id, public_id)
        )
        """
    )
    op.execute("CREATE INDEX agent_credentials_tenant_idx ON agent_credentials (tenant_id)")

    op.execute("ALTER TABLE agent_credentials ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE agent_credentials FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY agent_credentials_tenant_isolation ON agent_credentials
            USING      (tenant_id = current_setting('app.tenant_id', true)::uuid)
            WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid)
        """
    )

    # No DELETE grant -- see module docstring: revocation is the only lifecycle event this
    # table's grants permit short of tenant erasure.
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app') THEN
                GRANT SELECT, INSERT, UPDATE ON agent_credentials TO app;
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS agent_credentials")
