"""Repository layer: the only path to the data.

The session comes from tenant_session(ctx) and is therefore tenant-bound; RLS filters in the DB.
On writes, tenant_id is still set explicitly from the context (the policy's WITH CHECK rejects
anything else).
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.context import RequestContext
from app.db.models import Document


class DocumentHit(BaseModel):
    id: UUID
    title: str
    snippet: str
    score: float


class DocumentRepository:
    async def add(
        self,
        session: AsyncSession,
        ctx: RequestContext,
        *,
        title: str,
        content: str,
        embedding: list[float] | None,
        metadata: dict | None = None,
    ) -> Document:
        doc = Document(
            tenant_id=ctx.tenant_id,
            title=title,
            content=content,
            embedding=embedding,
            metadata_=metadata or {},
        )
        session.add(doc)
        await session.flush()
        return doc

    async def rename(
        self, session: AsyncSession, ctx: RequestContext, *, document_id: UUID, title: str
    ) -> Document | None:
        """Renames one of the tenant's documents -- the one write this template's example
        writing tool makes (ADR-0007, Spec 5 / #40), through the exact same repository layer
        `search` and `add` already use. None for an unknown document id or one belonging to
        another tenant (RLS plus the explicit WHERE clause make the two indistinguishable).
        `documents_set_update_audit` (migration 0010) refreshes `updated_by`/`updated_at` from
        `app.identity_id` the moment this UPDATE commits -- nothing here sets them explicitly."""
        doc = (
            await session.execute(
                select(Document).where(
                    Document.tenant_id == ctx.tenant_id, Document.id == document_id
                )
            )
        ).scalar_one_or_none()
        if doc is None:
            return None
        doc.title = title
        await session.flush()
        return doc

    async def search(
        self,
        session: AsyncSession,
        embedding: list[float],
        *,
        limit: int = 5,
        snippet_chars: int = 400,
    ) -> list[DocumentHit]:
        distance = Document.embedding.cosine_distance(embedding).label("distance")
        stmt = (
            select(Document, distance)
            .where(Document.embedding.is_not(None))
            .order_by(distance)
            .limit(limit)
        )
        rows = (await session.execute(stmt)).all()
        return [
            DocumentHit(
                id=doc.id,
                title=doc.title,
                snippet=doc.content[:snippet_chars],
                score=1.0 - float(dist),
            )
            for doc, dist in rows
        ]
