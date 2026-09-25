"""Conversation history loading — a thin wrapper around `ConversationsRepository` (ADR-0006,
Spec 4 / #33), shaped exactly like `app/tools/documents.py`'s `search_documents`: given a
context, open a tenant-bound session and go through the repository, returning only what is
needed. Used by `POST /v1/t/{tenant_id}/api/chat` to load the trusted, server-held history for
a conversation before running the agent — never anything from the request body itself.
"""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage

from app.context import RequestContext
from app.db.session import tenant_session
from app.repositories.conversations import ConversationsRepository


async def load_conversation_history(
    ctx: RequestContext, conversation_id: str
) -> list[ModelMessage]:
    """The ordered message history of `conversation_id`, scoped to the tenant (RLS) and to the
    member who started it. Empty — never an error — for an unknown conversation id or one
    started by a different member, per `ConversationsRepository.get_history`."""
    async with tenant_session(ctx) as session:
        return await ConversationsRepository().get_history(
            session, ctx, conversation_id=conversation_id
        )
