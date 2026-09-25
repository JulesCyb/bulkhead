"""Conversations repository (ADR-0006, Spec 4 / #32): the only path to `conversations` and
`messages`, shaped like `app/repositories/documents.py`.

The session comes from `tenant_session(ctx)` and is therefore tenant-bound; RLS filters both
tables to `ctx.tenant_id` before any code here runs (migration 0020). Member-level scoping --
"a conversation belongs to the member who started it" -- is *not* something RLS gives us (the
tenant policy only knows about `tenant_id`); it is enforced here, in application code, by
filtering on `created_by`.

Message payloads round-trip through pydantic-ai's own `ModelMessagesTypeAdapter`
(`pydantic_ai.messages`) rather than a hand-rolled shape, one row per `ModelMessage`
(`ModelRequest` or `ModelResponse`) -- exactly what a run's `result.new_messages()` returns.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.context import RequestContext
from app.db.models import Conversation, Message


class ConversationOwnershipError(PermissionError):
    """Raised when a caller tries to append to a conversation id already owned by a different
    member of the same tenant. Not exercised by the acceptance criteria directly, but a
    deliberate safety net: without it, two members choosing the same conversation id in the same
    tenant would silently share -- and each be able to write into -- one conversation."""


class ConversationsRepository:
    async def get_history(
        self, session: AsyncSession, ctx: RequestContext, *, conversation_id: str
    ) -> list[ModelMessage]:
        """The ordered message history of `conversation_id`, scoped to the tenant (via RLS) and
        to the member who started it (`created_by == ctx.identity_id`). Empty -- never an error
        -- for an unknown conversation id or one created by a different member, so a caller can
        never distinguish "does not exist" from "exists, but is not yours"."""
        conversation = await self._get_own_conversation(session, ctx, conversation_id)
        if conversation is None:
            return []

        rows = (
            await session.execute(
                select(Message.payload)
                .where(
                    Message.tenant_id == ctx.tenant_id,
                    Message.conversation_id == conversation_id,
                )
                .order_by(Message.sequence)
            )
        ).scalars()
        return [_load_message(payload) for payload in rows]

    async def append_run(
        self,
        session: AsyncSession,
        ctx: RequestContext,
        *,
        conversation_id: str,
        messages: list[ModelMessage],
    ) -> None:
        """Creates `conversation_id` if it does not exist yet, then appends `messages`, advancing
        the per-conversation sequence and the conversation's last-activity marker -- all within
        the caller's transaction (the session from `tenant_session(ctx)`), so a run that fails
        partway through leaves nothing half-written."""
        conversation = (
            await session.execute(
                select(Conversation).where(
                    Conversation.tenant_id == ctx.tenant_id,
                    Conversation.conversation_id == conversation_id,
                )
            )
        ).scalar_one_or_none()

        if conversation is None:
            conversation = Conversation(tenant_id=ctx.tenant_id, conversation_id=conversation_id)
            session.add(conversation)
            await session.flush()
        elif conversation.created_by != ctx.identity_id:
            raise ConversationOwnershipError(
                f"conversation {conversation_id!r} belongs to a different member"
            )

        if not messages:
            return

        next_sequence = (
            await session.execute(
                select(func.coalesce(func.max(Message.sequence), 0)).where(
                    Message.tenant_id == ctx.tenant_id,
                    Message.conversation_id == conversation_id,
                )
            )
        ).scalar_one()

        for offset, message in enumerate(messages, start=1):
            session.add(
                Message(
                    tenant_id=ctx.tenant_id,
                    conversation_id=conversation_id,
                    sequence=next_sequence + offset,
                    payload=_dump_message(message),
                )
            )
        await session.flush()
        # last_activity_at is advanced by the messages_touch_conversation trigger (migration
        # 0020), not by an UPDATE issued here -- app has no UPDATE grant on conversations at
        # all (the acceptance criteria's "exactly select/insert/delete"). Expire the in-memory
        # attribute so a caller that re-reads `conversation` in the same session sees the
        # trigger's write rather than a stale, pre-append value.
        session.expire(conversation, ["last_activity_at"])

    async def delete_expired(
        self, session: AsyncSession, ctx: RequestContext, *, older_than: datetime
    ) -> int:
        """Deletes `ctx.tenant_id`'s conversations whose `last_activity_at` is older than
        `older_than`, in one tenant-scoped statement (RLS already restricts it to `ctx.tenant_id`;
        no cross-tenant reach is possible even by mistake). Messages go with them via the
        cascading foreign key (migration 0020) -- no separate delete. Returns the number of
        conversations removed."""
        result = await session.execute(
            Conversation.__table__.delete()
            .where(
                Conversation.tenant_id == ctx.tenant_id,
                Conversation.last_activity_at < older_than,
            )
            .returning(Conversation.conversation_id)
        )
        return len(result.all())

    async def _get_own_conversation(
        self, session: AsyncSession, ctx: RequestContext, conversation_id: str
    ) -> Conversation | None:
        return (
            await session.execute(
                select(Conversation).where(
                    Conversation.tenant_id == ctx.tenant_id,
                    Conversation.conversation_id == conversation_id,
                    Conversation.created_by == ctx.identity_id,
                )
            )
        ).scalar_one_or_none()


def _dump_message(message: ModelMessage) -> dict[str, Any]:
    return ModelMessagesTypeAdapter.dump_python([message], mode="json")[0]


def _load_message(payload: dict[str, Any]) -> ModelMessage:
    return ModelMessagesTypeAdapter.validate_python([payload])[0]
