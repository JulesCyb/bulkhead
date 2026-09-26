"""Provider abstraction for language models.

Model name in PydanticAI format "<provider>:<model>" (LLM_MODEL). The gateway is a required
service, not an optional profile (ADR-0009): a deployment always routes through it
(OpenAI-compatible), which is what makes switching providers one config line. `Settings`
construction itself refuses to succeed with no `LITELLM_BASE_URL` configured
(`app.config.Settings._require_gateway_configured`) -- there is no direct-provider fallback
anywhere in this module, and there must never be one added. Per tenant,
tenants.settings["model"] can override the default (the model_name argument).

The one entry point that builds a chat client lives here:

- `resolve_tenant_chat_model()` — the per-tenant entry point (Spec 7 / #54, ADR-0009): before a
  chat client is ever built, the tenant's chosen model name is validated against the allow-list
  for its residency (`settings.residency_allow_list`, `app.residency.ResidencyAllowList`) and
  rejected with
  `ModelNotAllowedForResidency` if it is not on it — before any client is constructed and before
  any network call is attempted. The client that is finally built is constructed from that
  tenant's own gateway credential (`app.gateway_credentials`, #52) and cached per tenant id, so
  two tenants never share a connection; every call it makes carries an explicit wall-clock
  deadline (`model_settings.timeout`) instead of the client library's multi-minute default.

There used to be a second entry point here, `get_model()` — a deployment-wide resolver with its
own, separate `LITELLM_BASE_URL` check that fell back to returning a bare `"<provider>:<model>"`
string (a direct-provider call, bypassing the gateway entirely) when no gateway was configured.
It was removed (ai-app-starter#7 review finding, ADR-0009): `Settings` now refuses to construct
without a gateway URL at all, so that fallback branch could never legitimately run, and its mere
presence was a second, unsupervised way for future code to skip the gateway. Do not re-add a
function that builds a model/client from `Settings` without going through
`resolve_tenant_chat_model()` (or `build_tenant_chat_model()`, which it calls) — both require a
gateway credential and a configured `litellm_base_url`, by construction.
"""

from __future__ import annotations

from uuid import UUID

import httpx
from pydantic import SecretStr
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.context import RequestContext
from app.gateway_credentials import resolve_gateway_credential
from app.repositories.control import ControlRepository
from app.residency import ResidencyUnresolved


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
            f"(see settings.residency_allow_list, app.residency.ResidencyAllowList)"
        )


def _bare_model_name(name: str) -> str:
    """Strips a "<provider>:" prefix, if any -- the gateway's model_list only knows bare names."""
    return name.split(":", 1)[1] if ":" in name else name


def validate_model_for_residency(
    model_name: str, residency: str, *, settings: Settings | None = None
) -> str:
    """Returns the bare (gateway alias) model name if `model_name` is allow-listed for
    `residency`; raises `ModelNotAllowedForResidency` otherwise. Synchronous and side-effect-free
    beyond reading `settings.residency_allow_list` (`get_settings()` when not given) -- no client,
    no network call, no database.
    """
    s = settings or get_settings()
    bare = _bare_model_name(model_name)
    allowed = s.residency_allow_list.model_aliases(residency)  # type: ignore[union-attr]
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

    In order: resolves the tenant's residency from the control plane (raising `ResidencyUnresolved`
    if none is recorded -- ADR-0008 fails closed, no default), validates the requested model name
    (or the deployment default) against that residency's allow-list -- raising
    `ModelNotAllowedForResidency` before anything else happens -- then resolves the tenant's own
    gateway credential and builds (or reuses) its cached client.
    """
    s = settings or get_settings()
    residency = await ControlRepository().get_residency(session, ctx)
    if residency is None:
        raise ResidencyUnresolved(f"tenant {ctx.tenant_id} has no residency recorded")
    name = model_name or s.llm_model
    bare = validate_model_for_residency(name, residency, settings=s)
    credential = await resolve_gateway_credential(session, ctx, settings=s)
    return build_tenant_chat_model(ctx.tenant_id, bare, credential, settings=s)


__all__ = [
    "ModelNotAllowedForResidency",
    "build_tenant_chat_model",
    "reset_tenant_chat_model_cache",
    "resolve_tenant_chat_model",
    "validate_model_for_residency",
]
