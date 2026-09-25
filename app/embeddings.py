"""Embeddings via an OpenAI-compatible endpoint (direct or through LiteLLM).

The dimension must match the documents.embedding column (migration 0001: 1536).
"""

from __future__ import annotations

from functools import lru_cache

from openai import AsyncOpenAI

from app.config import get_settings


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
