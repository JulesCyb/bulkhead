"""Provider abstraction for language models.

Model name in PydanticAI format "<provider>:<model>" (LLM_MODEL). If LITELLM_BASE_URL is set,
everything goes through the LiteLLM gateway (OpenAI-compatible) — switching providers is one
config line.
Per tenant, tenants.settings["model"] can override the default (the model_name argument).
"""

from __future__ import annotations

from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.config import get_settings


def get_model(model_name: str | None = None) -> Model | str:
    s = get_settings()
    name = model_name or s.llm_model
    if s.litellm_base_url:
        # Through the gateway: the bare model name from litellm/config.yaml, no provider prefix.
        bare = name.split(":", 1)[1] if ":" in name else name
        api_key = s.litellm_api_key.get_secret_value() if s.litellm_api_key else "litellm"
        provider = OpenAIProvider(base_url=s.litellm_base_url, api_key=api_key)
        return OpenAIChatModel(bare, provider=provider)
    return name
