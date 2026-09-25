"""Tool functions for documents — a shared module for agent tools AND the MCP server.

Rule: a tool receives the context, opens its own tenant-bound session, goes through the
repository, and returns only what is needed (a snippet, not the full text). Everything returned
here ends up in the prompt sent to the model provider.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.context import RequestContext
from app.db.session import tenant_session
from app.embeddings import embed
from app.repositories.documents import DocumentHit, DocumentRepository


async def search_documents(ctx: RequestContext, query: str, limit: int = 5) -> list[DocumentHit]:
    """Semantic search in the tenant's documents (RLS filters in the DB)."""
    limit = max(1, min(limit, 20))
    embedding = await embed(query)
    async with tenant_session(ctx) as session:
        return await DocumentRepository().search(session, embedding, limit=limit)


class DocumentRenamed(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str


async def rename_document(
    ctx: RequestContext, *, document_id: UUID, title: str
) -> DocumentRenamed | None:
    """Renames one of the tenant's documents through `DocumentRepository.rename` -- the example
    writing tool ADR-0007 and Spec 5 (#40) call for: the same repository layer every reading tool
    already uses, never a connection or credential of its own. No role or approval check here --
    that machinery (`app/tools/approvals.py`) lives one layer up, wrapped around this function by
    `app/agents/assistant.py`'s writing-tool registration, exactly the seam CLAUDE.md rule 4 asks
    a derived project's own first writing tool to reuse."""
    async with tenant_session(ctx) as session:
        doc = await DocumentRepository().rename(session, ctx, document_id=document_id, title=title)
        return DocumentRenamed.model_validate(doc) if doc is not None else None
