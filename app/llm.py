"""Provider abstraction for language models.

Model name in PydanticAI format "<provider>:<model>" (LLM_MODEL). If LITELLM_BASE_URL is set,
everything goes through the LiteLLM gateway (OpenAI-compatible) — switching providers is one
config line.
Per tenant, tenants.settings["model"] can override the default (the model_name argument).

Two entry points live here:

- `get_model()` — the pre-existing, deployment-wide resolver: no tenant, no residency check, no
  per-tenant credential. Kept for the deployment default and for tests that don't need a tenant.
- `resolve_tenant_chat_model()` — the per-tenant entry point (Spec 7 / #54, ADR-0009): before a
  chat client is ever built, the tenant's chosen model name is validated against the allow-list
  for its residency (`RESIDENCY_MODEL_ALLOW_LIST`, app/config.py) and rejected with
  `ModelNotAllowedForResidency` if it is not on it — before any client is constructed and before
  any network call is attempted. The client that is finally built is constructed from that
  tenant's own gateway credential (`app.gateway_credentials`, #52) and cached per tenant id, so
  two tenants never share a connection; every call it makes carries an explicit wall-clock
  deadline (`model_settings.timeout`) instead of the client library's multi-minute default.
"""

from __future__ import annotations

from uuid import UUID

import httpx
from pydantic import SecretStr
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import RESIDENCY_MODEL_ALLOW_LIST, Settings, get_settings
from app.context import RequestContext
from app.gateway_credentials import resolve_gateway_credential
from app.repositories.control import ControlRepository


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


class ModelNotAllowedForResidency(Exception):
    """A tenant's chosen model name is not on the allow-list for its residency (ADR-0009).

    Raised before any client is constructed and before any network call is attempted -- the
    application's own half of ADR-0009's two-layer defense; the gateway's own key-scoped models
    are the second layer.
    """

    def __init__(self, model_name: str, residency: str) -> None:
        self.model_name = model_name
        self.residency = residency
        super().__init__(
            f"model {model_name!r} is not on the allow-list for residency {residency!r} "
            f"(see RESIDENCY_MODEL_ALLOW_LIST in app/config.py)"
        )


def _bare_model_name(name: str) -> str:
    """Strips a "<provider>:" prefix, if any -- the gateway's model_list only knows bare names."""
    return name.split(":", 1)[1] if ":" in name else name


def validate_model_for_residency(model_name: str, residency: str) -> str:
    """Returns the bare (gateway alias) model name if `model_name` is allow-listed for
    `residency`; raises `ModelNotAllowedForResidency` otherwise. Pure and synchronous: no client,
    no network call, no database -- callable directly against configuration and a residency
    string, before anything else is resolved.
    """
    bare = _bare_model_name(model_name)
    allowed = RESIDENCY_MODEL_ALLOW_LIST.get(residency, ())
    if bare not in allowed:
        raise ModelNotAllowedForResidency(model_name, residency)
    return bare


# Cache of per-tenant chat model clients (ADR-0009: "any cache of them is keyed by tenant"), so
# two tenants never share a connection or credential, and resolving the same tenant twice reuses
# one client rather than reconnecting. Never keyed by anything else (not by model name): a
# tenant's credential -- not its momentary model choice -- is the resource worth not duplicating.
_tenant_chat_models: dict[UUID, OpenAIChatModel] = {}


def build_tenant_chat_model(
    tenant_id: UUID,
    bare_model_name: str,
    credential: SecretStr,
    *,
    settings: Settings | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> OpenAIChatModel:
    """Builds (or reuses) the chat model client for `tenant_id`, from that tenant's own gateway
    credential -- never a process-wide key. Cached per tenant id: calling this twice for the same
    `tenant_id` returns the exact same object; calling it for a different `tenant_id` always
    returns a distinct one. Carries an explicit wall-clock deadline
    (`ModelSettings.timeout` = `Settings.llm_call_timeout_seconds`), replacing the client
    library's multi-minute default.

    `http_client` is test-only (an already-configured `httpx.AsyncClient`, e.g. one backed by a
    `MockTransport`); production call sites never pass it.
    """
    cached = _tenant_chat_models.get(tenant_id)
    if cached is not None:
        return cached
    s = settings or get_settings()
    if not s.litellm_base_url:
        raise RuntimeError(
            "LITELLM_BASE_URL must be configured to build a per-tenant model client: a gateway "
            "credential is only valid against the gateway, never a raw provider endpoint."
        )
    provider = OpenAIProvider(
        base_url=s.litellm_base_url,
        api_key=credential.get_secret_value(),
        http_client=http_client,
    )
    model = OpenAIChatModel(
        bare_model_name,
        provider=provider,
        settings=ModelSettings(timeout=s.llm_call_timeout_seconds),
    )
    _tenant_chat_models[tenant_id] = model
    return model


def reset_tenant_chat_model_cache() -> None:
    """Test-only: clears the per-tenant chat model cache between test cases."""
    _tenant_chat_models.clear()


async def resolve_tenant_chat_model(
    session: AsyncSession,
    ctx: RequestContext,
    model_name: str | None = None,
    *,
    settings: Settings | None = None,
) -> OpenAIChatModel:
    """The per-tenant model-resolution entry point (ADR-0009, Spec 7 / #54).

    In order: resolves the tenant's residency from the control plane (falling back to the
    deployment's own `Settings.residency` if the tenant has no control-plane row yet, mirroring
    `resolve_gateway_credential_alias`'s fallback pattern), validates the requested model name
    (or the deployment default) against that residency's allow-list -- raising
    `ModelNotAllowedForResidency` before anything else happens -- then resolves the tenant's own
    gateway credential and builds (or reuses) its cached client.
    """
    s = settings or get_settings()
    residency = await ControlRepository().get_residency(session, ctx) or s.residency
    name = model_name or s.llm_model
    bare = validate_model_for_residency(name, residency)
    credential = await resolve_gateway_credential(session, ctx, settings=s)
    return build_tenant_chat_model(ctx.tenant_id, bare, credential, settings=s)


__all__ = [
    "ModelNotAllowedForResidency",
    "build_tenant_chat_model",
    "get_model",
    "reset_tenant_chat_model_cache",
    "resolve_tenant_chat_model",
    "validate_model_for_residency",
]
