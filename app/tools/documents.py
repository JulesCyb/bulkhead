"""Tool functions for documents — a shared module for agent tools AND the MCP server.

Rule: a tool receives the context, opens its own tenant-bound session, goes through the
repository, and returns only what is needed (a snippet, not the full text). Everything returned
here ends up in the prompt sent to the model provider.

The search query is embedded through `app.embeddings.resolve_tenant_embedding_client` (Spec 8 /
#61, ADR-0008, #105), a function of the tenant record: the endpoint it embeds against is resolved
from the requesting tenant's own residency, never a process-wide default. A tenant with no
resolvable residency fails closed (`app.residency.ResidencyUnresolved`) here exactly as it does on
the chat path, rather than silently reaching some default embedding provider.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.config import get_settings
from app.context import RequestContext
from app.db.session import tenant_session
from app.embeddings import resolve_tenant_embedding_client
from app.repositories.documents import DocumentHit, DocumentRepository
from app.tenant_record import TenantRecord
from app.token_verifier import default_adapter


async def _tenant_record(ctx: RequestContext) -> TenantRecord:
    """The record the context carries (every HTTP request and streamable-http MCP connection:
    read once at context resolution, #104). Only a context built without that step -- the stdio
    MCP development fallback (`app.context_resolution.resolve_stdio_env_context`) -- has none; it
    gets the record through the very same record read (`ControlPlaneReads.get_tenant_record`),
    never a residency read of its own."""
    if ctx.tenant_record is not None:
        return ctx.tenant_record
    return await default_adapter().get_tenant_record(tenant_id=ctx.tenant_id)


async def search_documents(ctx: RequestContext, query: str, limit: int = 5) -> list[DocumentHit]:
    """Semantic search in the tenant's documents (RLS filters in the DB).

    The embedding client (residency route + gateway credential) is resolved from the tenant
    record -- no database read of its own (#105); the one tenant-bound session is for the
    RLS-filtered document search itself.
    """
    limit = max(1, min(limit, 20))
    async with tenant_session(ctx) as session:
        client = resolve_tenant_embedding_client(await _tenant_record(ctx))
        settings = get_settings()
        response = await client.embeddings.create(
            model="embeddings", input=query, dimensions=settings.embedding_dimensions
        )
        embedding = list(response.data[0].embedding)
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
