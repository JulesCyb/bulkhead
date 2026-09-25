"""Conversations and messages: server-side chat history under RLS (ADR-0006, Spec 4 / #32).

Revision ID: 0020
Revises: 0016
Create Date: 2026-09-25

`conversations` and `messages` get the same four mandatory parts every table in this project
gets: `tenant_id NOT NULL REFERENCES tenants(id)`, an index on it, `ENABLE`/`FORCE ROW LEVEL
SECURITY`, and the standard tenant-isolation policy (USING and WITH CHECK on
`current_setting('app.tenant_id', true)`), plus a grant to the `app` role for *exactly*
`SELECT, INSERT, DELETE` on both tables -- no `UPDATE`, matching the acceptance criteria and the
spec's own "read, insert, and the delete the retention job needs -- no more" (Implementation
Decisions). That leaves one problem: appending a run's messages must also advance
`conversations.last_activity_at` in the same transaction, and `app` has no `UPDATE` grant to do
that itself. This is solved the same way 0016 solved an equivalent "app needs a narrow escape
hatch it must never hold generally" problem: a `SECURITY DEFINER` trigger function, owned by
`app_owner` (never granted to `app` directly, and not invokable except by firing the trigger),
that runs the one `UPDATE` the repository is allowed to cause -- refreshing
`last_activity_at = now()` on the parent conversation -- every time a message is inserted. `app`
still cannot run `UPDATE conversations` itself; it can only cause this one, narrow side effect by
doing the `INSERT` it is already allowed to do.

`conversations` is keyed by the pair of tenant and the client's own conversation id (the Vercel
chat `id`), not a server-generated one -- the client names a conversation and the server
recognizes it on the next request; a second tenant choosing the same id is a different row, never
a collision (Spec 4 Implementation Decisions). `created_by` follows the audit-column pattern
migration 0010 established for `documents` verbatim: a `NOT NULL` foreign key to
`control.identities`, defaulted from the per-transaction `app.identity_id` setting
(`tenant_session()` in `app/db/session.py`) -- never passed explicitly by the repository.

`messages` carries a monotonically increasing `sequence` within its conversation (application-
assigned, `UNIQUE (tenant_id, conversation_id, sequence)` catches a race rather than silently
corrupting order) because a reload must reconstruct the exact order a run originally produced --
timestamp ties are not a safe substitute (Spec 4, story 20/21). `payload` is one library-native
`pydantic_ai.messages` message (a `ModelRequest` or `ModelResponse`), serialized with the
library's own `ModelMessagesTypeAdapter`, one row per message -- never a hand-rolled shape
(Spec 4, story 21; `app/repositories/conversations.py`).

Both tables reference `tenants(id) ON DELETE CASCADE`, and `messages` additionally references its
own conversation `ON DELETE CASCADE` -- deleting a tenant, or the retention job deleting one
tenant's expired conversation rows, removes the matching messages for free, no separate cleanup
step (Spec 4 Implementation Decisions: "Tenant erasure removes conversations for free").
"""

from collections.abc import Sequence

from alembic import op

_TABLES_AT_THIS_REVISION: tuple[str, ...] = ("conversations", "messages")

revision: str = "0020"
down_revision: str | None = "0016"
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
    op.execute(
        """
        CREATE TABLE conversations (
            tenant_id         uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            conversation_id   varchar(200) NOT NULL,
            created_by        uuid NOT NULL
                                  DEFAULT current_setting('app.identity_id', true)::uuid
                                  REFERENCES control.identities (id),
            created_at        timestamptz NOT NULL DEFAULT now(),
            last_activity_at  timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, conversation_id)
        )
        """
    )
    op.execute("CREATE INDEX conversations_tenant_idx ON conversations (tenant_id)")

    op.execute(
        """
        CREATE TABLE messages (
            id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id        uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            conversation_id  varchar(200) NOT NULL,
            sequence         integer NOT NULL,
            payload          jsonb NOT NULL,
            created_by       uuid NOT NULL
                                 DEFAULT current_setting('app.identity_id', true)::uuid
                                 REFERENCES control.identities (id),
            created_at       timestamptz NOT NULL DEFAULT now(),
            FOREIGN KEY (tenant_id, conversation_id)
                REFERENCES conversations (tenant_id, conversation_id) ON DELETE CASCADE,
            UNIQUE (tenant_id, conversation_id, sequence)
        )
        """
    )
    op.execute("CREATE INDEX messages_tenant_idx ON messages (tenant_id)")

    for table in _TABLES_AT_THIS_REVISION:
        _rls(table)

    # The narrow escape hatch described in the module docstring: app_owner-owned, SECURITY
    # DEFINER, so it can UPDATE conversations.last_activity_at even though app itself is never
    # granted UPDATE on that table -- the only way to cause this side effect is to INSERT a
    # message, which app is already allowed to do.
    op.execute(
        """
        CREATE FUNCTION conversations_touch_last_activity() RETURNS trigger
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = public, pg_temp
        AS $$
        BEGIN
            UPDATE conversations
                SET last_activity_at = now()
                WHERE tenant_id = NEW.tenant_id AND conversation_id = NEW.conversation_id;
            RETURN NEW;
        END;
        $$
        """
    )
    op.execute("REVOKE ALL ON FUNCTION conversations_touch_last_activity() FROM PUBLIC")
    op.execute(
        """
        CREATE TRIGGER messages_touch_conversation
            AFTER INSERT ON messages
            FOR EACH ROW
            EXECUTE FUNCTION conversations_touch_last_activity()
        """
    )

    # Grants for the app role (exists only if 01-init.sh has run -- skip otherwise, matching
    # 0001's pattern): exactly SELECT, INSERT, DELETE on both tables -- no UPDATE, per the
    # acceptance criteria and the module docstring above.
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app') THEN
                GRANT SELECT, INSERT, DELETE ON conversations, messages TO app;
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS messages_touch_conversation ON messages")
    op.execute("DROP FUNCTION IF EXISTS conversations_touch_last_activity()")
    op.execute("DROP TABLE IF EXISTS messages")
    op.execute("DROP TABLE IF EXISTS conversations")
