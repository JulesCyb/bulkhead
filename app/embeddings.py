"""Embeddings via the LiteLLM gateway's OpenAI-compatible endpoint (required, ADR-0009).

The dimension must match the documents.embedding column (migration 0001: 1536).

Two entry points: `embed()` — the pre-existing, deployment-wide client, kept for call sites and
tests with no tenant context — and `resolve_tenant_embedding_client()` (Spec 7 / #54, ADR-0009):
built from a tenant's own gateway credential, cached per tenant id so two tenants never share a
connection, and carrying an explicit wall-clock deadline instead of the client library's
multi-minute default. The embedding model itself is not tenant-choosable (ADR-0008: one embedding
model family is shared across every residency, since the vector column's dimension is fixed), so
there is no allow-list check here — that only applies to the chat model (`app/llm.py`).
"""

from __future__ import annotations

from functools import lru_cache
from uuid import UUID

import httpx
from openai import AsyncOpenAI
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.context import RequestContext
from app.gateway_credentials import resolve_gateway_credential


@lru_cache
def _client() -> AsyncOpenAI:
    # One client per process: AsyncOpenAI holds an httpx connection pool; constructing it
    # per call would leak connections and pay TCP+TLS setup on every search.
    s = get_settings()
    if s.litellm_base_url:
        api_key = s.litellm_api_key.get_secret_value() if s.litellm_api_key else "litellm"
        return AsyncOpenAI(base_url=s.litellm_base_url, api_key=api_key)
    return AsyncOpenAI(api_key=s.openai_api_key.get_secret_value() if s.openai_api_key else None)


async def embed(text: str) -> list[float]:
    s = get_settings()
    response = await _client().embeddings.create(
        model=s.embedding_model, input=text, dimensions=s.embedding_dimensions
    )
    return list(response.data[0].embedding)


# Cache of per-tenant embedding clients (ADR-0009: "any cache of them is keyed by tenant"), the
# embedding-side counterpart of app.llm's `_tenant_chat_models` -- same reasoning, same shape.
_tenant_embedding_clients: dict[UUID, AsyncOpenAI] = {}


def build_tenant_embedding_client(
    tenant_id: UUID,
    credential: SecretStr,
    *,
    settings: Settings | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> AsyncOpenAI:
    """Builds (or reuses) the embedding client for `tenant_id`, from that tenant's own gateway
    credential. Cached per tenant id: calling this twice for the same `tenant_id` returns the
    exact same object; a different `tenant_id` always returns a distinct one. Carries an explicit
    wall-clock deadline (`Settings.embedding_call_timeout_seconds`), replacing the client
    library's multi-minute default.

    `http_client` is test-only; production call sites never pass it.
    """
    cached = _tenant_embedding_clients.get(tenant_id)
    if cached is not None:
        return cached
    s = settings or get_settings()
    if not s.litellm_base_url:
        raise RuntimeError(
            "LITELLM_BASE_URL must be configured to build a per-tenant embedding client: a "
            "gateway credential is only valid against the gateway, never a raw provider endpoint."
        )
    client = AsyncOpenAI(
        base_url=s.litellm_base_url,
        api_key=credential.get_secret_value(),
        timeout=s.embedding_call_timeout_seconds,
        http_client=http_client,
    )
    _tenant_embedding_clients[tenant_id] = client
    return client


def reset_tenant_embedding_client_cache() -> None:
    """Test-only: clears the per-tenant embedding client cache between test cases."""
    _tenant_embedding_clients.clear()


async def resolve_tenant_embedding_client(
    session: AsyncSession, ctx: RequestContext, *, settings: Settings | None = None
) -> AsyncOpenAI:
    """Resolves `ctx.tenant_id`'s own gateway credential and builds (or reuses) its cached
    embedding client -- the embedding-side counterpart of `app.llm.resolve_tenant_chat_model`."""
    s = settings or get_settings()
    credential = await resolve_gateway_credential(session, ctx, settings=s)
    return build_tenant_embedding_client(ctx.tenant_id, credential, settings=s)
