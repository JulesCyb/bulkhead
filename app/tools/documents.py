"""Tool functions for documents — a shared module for agent tools AND the MCP server.

Rule: a tool receives the context, opens its own tenant-bound session, goes through the
repository, and returns only what is needed (a snippet, not the full text). Everything returned
here ends up in the prompt sent to the model provider.

The search query is embedded through `app.embeddings.resolve_tenant_embedding_client` (Spec 8 /
#61, ADR-0008): the endpoint it embeds against is resolved from the requesting tenant's own
residency, never a process-wide default. A tenant with no resolvable residency fails closed
(`app.residency.ResidencyUnresolved`) here exactly as it does on the chat path, rather than
silently reaching some default embedding provider.
"""

from __future__ import annotations

from app.config import get_settings
from app.context import RequestContext
from app.db.session import tenant_session
from app.embeddings import resolve_tenant_embedding_client
from app.repositories.documents import DocumentHit, DocumentRepository


async def search_documents(ctx: RequestContext, query: str, limit: int = 5) -> list[DocumentHit]:
    """Semantic search in the tenant's documents (RLS filters in the DB).

    Reuses the one tenant-bound session both for resolving the tenant's own embedding client
    (residency + gateway credential) and for the RLS-filtered document search itself.
    """
    limit = max(1, min(limit, 20))
    async with tenant_session(ctx) as session:
        client = await resolve_tenant_embedding_client(session, ctx)
        settings = get_settings()
        response = await client.embeddings.create(
            model="embeddings", input=query, dimensions=settings.embedding_dimensions
        )
        embedding = list(response.data[0].embedding)
        return await DocumentRepository().search(session, embedding, limit=limit)
