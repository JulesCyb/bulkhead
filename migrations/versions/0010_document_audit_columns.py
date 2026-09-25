"""Audit columns on `documents`: who created it, who last touched it (Spec 3 / #29).

Revision ID: 0010
Revises: 0008
Create Date: 2026-09-25

This is the audit-column pattern every later tenant table should copy verbatim: two columns,
`created_by`/`updated_by`, both `NOT NULL` foreign keys to `control.identities` -- never to a
membership, because a membership can be revoked while the document it produced still needs an
attributable author (ADR-0003's identity/membership split; a membership's role or even its
existence is orthogonal to who wrote a piece of content). Both columns are derived by the
database itself from the per-transaction `app.identity_id` setting that `tenant_session()`
already sets on every request (`app/db/session.py`) -- never from anything the application
passes explicitly, and never from anything a client could put in a request body:

- `created_by` gets a column `DEFAULT` reading `current_setting('app.identity_id', true)::uuid`,
  so a plain `INSERT` that never mentions the column still gets the right value, exactly the way
  `id` already defaults to `gen_random_uuid()`.
- A `DEFAULT` only fires on `INSERT`, so `updated_by` (and `updated_at`, added alongside it --
  a "last touched by" column is meaningless without a "last touched when" one) needs a
  `BEFORE UPDATE` trigger instead. The trigger also re-asserts `created_by` from the row's own
  previous value on every update, so an `UPDATE ... SET created_by = ...` from the application
  (there is no such call today, and there should never be one) could not silently rewrite
  history even if one were added by mistake later.

The foreign keys point at `control.identities`, whose only grant to the `app` role is the
narrow, issuer-plus-subject lookup view from ADR-0003 (`control.identity_lookup`, `SELECT`
only) -- `app` has no grant whatsoever on `control.identities` itself. This still works, and is
not an oversight: Postgres checks a foreign key constraint using the *referenced* table's
owner's privileges, not the querying role's (see the Postgres documentation on `REFERENCES`
privileges -- the constraint's owner, here `app_owner`, is who must be able to see the row being
referenced, and `app_owner` created `control.identities` and needs no extra grant on its own
table). No grant on `control.identities` is added here, and none is needed for `app`'s inserts
and updates on `documents` to have their `created_by`/`updated_by` values validated against it.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0010"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # NOT NULL + DEFAULT in the same ALTER TABLE: on a fresh install (the only case this
    # template ships against) `documents` is empty, so there are no existing rows for the
    # default to backfill. A derived project applying this migration against a table that
    # already has rows must backfill created_by/updated_by itself first.
    op.execute(
        """
        ALTER TABLE documents
            ADD COLUMN created_by uuid NOT NULL
                DEFAULT current_setting('app.identity_id', true)::uuid
                REFERENCES control.identities (id)
        """
    )
    op.execute(
        """
        ALTER TABLE documents
            ADD COLUMN updated_by uuid NOT NULL
                DEFAULT current_setting('app.identity_id', true)::uuid
                REFERENCES control.identities (id)
        """
    )
    op.execute("ALTER TABLE documents ADD COLUMN updated_at timestamptz NOT NULL DEFAULT now()")

    op.execute(
        """
        CREATE FUNCTION documents_set_update_audit() RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            NEW.created_by := OLD.created_by;
            NEW.updated_by := current_setting('app.identity_id', true)::uuid;
            NEW.updated_at := now();
            RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER documents_set_update_audit
            BEFORE UPDATE ON documents
            FOR EACH ROW
            EXECUTE FUNCTION documents_set_update_audit()
        """
    )
    # No new grant: documents already grants app SELECT/INSERT/UPDATE/DELETE on the whole row
    # (migration 0001), which covers the two new columns; no grant on control.identities is
    # needed either, per the module docstring above.


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS documents_set_update_audit ON documents")
    op.execute("DROP FUNCTION IF EXISTS documents_set_update_audit()")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS updated_at")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS updated_by")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS created_by")
