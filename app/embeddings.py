"""Embeddings via the LiteLLM gateway's OpenAI-compatible endpoint (required, ADR-0009;
Spec 8 / #61, ADR-0008).

The dimension must match the documents.embedding column (migration 0001: 1536).

The single entry point is `resolve_tenant_embedding_client()` (Spec 7 / #54, Spec 8 / #61,
ADR-0009, ADR-0008): it resolves `ctx.tenant_id`'s residency route and gateway credential
together through `app.residency.resolve_residency_route` -- the one call site both this module
and `app/llm.py` use, so a tenant with no resolvable residency fails closed
(`app.residency.ResidencyUnresolved`) exactly the same way on the embedding path as on the chat
path, rather than reaching some process-wide default endpoint. The client built from that route
is cached per tenant id so two tenants never share a connection, and carries an explicit
wall-clock deadline instead of the client library's multi-minute default. There used to be a
second, deployment-wide entry point (`embed()`) that could reach a raw OpenAI endpoint directly,
bypassing the gateway and any residency check; it has been removed (#61) so no code path can
still reach a hard-coded default embedding endpoint.

Every residency shares one embedding model family -- residency only ever selects the serving
region an embedding call is routed to, never a different model. This is fixed by the
`documents.embedding` column's dimension (pgvector indexes need a fixed dimension across every
row, whichever residency wrote them): adding a second embedding model family is not a
configuration change here, it is a new vector column and a re-embedding migration for every
existing row. The embedding model itself is therefore not tenant-choosable and carries no
allow-list check here -- that only applies to the chat model (`app/llm.py`).
"""

from __future__ import annotations

from uuid import UUID

import httpx
from openai import AsyncOpenAI
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.context import RequestContext
from app.residency import resolve_residency_route

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
    """The per-tenant embedding entry point (Spec 8 / #61, ADR-0008) -- the embedding-side
    counterpart of `app.llm.resolve_tenant_chat_model`.

    Resolves `ctx.tenant_id`'s residency route through `app.residency.resolve_residency_route`
    (the same single call site the chat path's residency check ultimately rests on), then builds
    (or reuses) the tenant's own cached embedding client from the credential that route bundles.
    Raises `app.residency.ResidencyUnresolved` if the tenant's residency is missing, unknown, or
    absent from the allow-list -- exactly the same fail-closed behaviour as the chat path, never a
    fallback to a default embedding endpoint -- and propagates
    `app.gateway_credentials.GatewayCredentialUnavailable` unchanged if the credential itself
    cannot be resolved.
    """
    s = settings or get_settings()
    resolved = await resolve_residency_route(session, ctx, settings=s)
    return build_tenant_embedding_client(ctx.tenant_id, resolved.gateway_credential, settings=s)
