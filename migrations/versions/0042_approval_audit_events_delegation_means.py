"""Persist the delegation means (ADR-0005) on every `approval_audit_events` row (#117).

Revision ID: 0042
Revises: 0041
Create Date: 2026-09-27

`approval_audit_events` (migration 0035) already records the *approval* means -- which
`pending_action_id`/`standing_grant_id` (if either) a milestone came from -- and the acting
membership. It said nothing about the *delegation* means (ADR-0005 / CONTEXT.md's "Delegation"
glossary entry): whether the request that produced the milestone ran as a person acting through
the assistant (`("agent", "assistant")`) or an agent identity acting on its own credential
(`("credential", <the credential's public id>)`). `RequestContext.means` has carried that fact
since #101, but it reached only trace spans, never a table a caller could read back.

**Two senses of "means" live on this one row, and must stay distinct.** The *approval* means
(`pending_action_id`/`standing_grant_id`, added in 0035) answers "which specific pending action or
standing grant authorized this write". The *delegation* means (`means_kind`/`means_id`, added
here) answers a different question entirely: "who -- or what -- was actually driving the request
that produced this milestone, as opposed to which record permitted it". Neither is derived from
the other, and a row can carry one, the other, both, or (a `denied_for_lack_of_grant` milestone,
by definition) neither approval-means value while still carrying its delegation means, because the
delegation means is a fact about the *request*, not about whatever pending action or standing
grant it happens to reference.

`means_kind` is restricted by a `CHECK` constraint to the exact two `MeansKind` literals
`app/context.py` defines (`"agent"`, `"credential"`) -- mirroring how `kind` itself is restricted
to the seven milestone literals in 0035 -- so the column can never silently drift from the type it
mirrors. Both columns are nullable: a context with no means attached (a test built without
`means=`, the `stdio` development fallback, which ADR-0005 says carries no means to report at all)
writes both null, which is the correct, unremarkable value there, never an error.

No other property of the table changes. RLS stays exactly as 0035 left it (`ENABLE`/`FORCE`, the
same tenant-isolation policy, unaffected by adding two plain columns) and the `app` role's grant
stays `SELECT, INSERT` only -- adding a column is not an excuse to add `UPDATE`/`DELETE` to an
append-only table (CLAUDE.md rule 2 / migration 0035's own docstring). No new index: nothing reads
this table by `means_kind`/`means_id` today, and the ticket does not ask for one.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0042"
down_revision: str | None = "0041"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE approval_audit_events
            ADD COLUMN means_kind varchar(20)
                CHECK (means_kind IN ('agent', 'credential')),
            ADD COLUMN means_id varchar(200)
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE approval_audit_events
            DROP COLUMN IF EXISTS means_kind,
            DROP COLUMN IF EXISTS means_id
        """
    )
