"""Provider abstraction for language models.

Model name in PydanticAI format "<provider>:<model>" (LLM_MODEL). The gateway is a required
service, not an optional profile (ADR-0009): a deployment always routes through it
(OpenAI-compatible), which is what makes switching providers one config line. `Settings`
construction itself refuses to succeed with no `LITELLM_BASE_URL` configured
(`app.config.Settings._require_gateway_configured`) -- there is no direct-provider fallback
anywhere in this module, and there must never be one added. Per tenant,
tenants.settings["model"] overrides the default -- read from the tenant record
(`app.tenant_record.TenantRecord.settings.model`, #105), validated on write by the operator
tool's `create` (`app.operator.create._validate_model`) and again here on every read.

The one entry point that builds a chat client lives here:

- `resolve_tenant_chat_model(record, *, settings)` — the per-tenant entry point (Spec 7 / #54,
  ADR-0009), a function of the tenant record and settings with no database read of its own
  (#105): before a chat client is ever built, the tenant's chosen model name (its own
  `model` setting, else `Settings.llm_model`) is validated against the allow-list
  for its residency (`validate_model_for_residency`, a thin call to
  `settings.residency_allow_list.alias_for` -- `app.residency.ResidencyAllowList`, spec A4 /
  #111) and rejected with `ModelNotAllowedForResidency` if it is not on it — before any client is
  constructed and before any network call is attempted. `ModelNotAllowedForResidency` is a
  subclass of `app.residency.ResidencyUnresolved` (not a second, unrelated exception type): a
  caller that only ever wants to fail closed on "this allow-list cannot answer that" catches the
  one base type, while this name is kept because `CLAUDE.md` and the tests still name it. The
  client that is finally built is constructed from that tenant's own gateway credential
  (`app.gateway_credentials`, #52, the alias the record carries); its connection is cached per
  tenant id, so two tenants never share one; every call it makes carries an explicit wall-clock
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

from app.config import Settings, get_settings
from app.gateway_credentials import resolve_gateway_credential
from app.residency import ResidencyUnresolved, tenant_residency_route
from app.tenant_record import TenantRecord


class ModelNotAllowedForResidency(ResidencyUnresolved):
    """A tenant's chosen model name is not on the allow-list for its residency (ADR-0009).

    A subclass of `app.residency.ResidencyUnresolved` (spec A4 / #111) -- both are "this
    allow-list cannot answer that", one from a specific model choice, the other from a residency
    or route lookup -- rather than a second, unrelated exception type; a caller that only ever
    wants to fail closed can catch the one base type. Kept as its own name (with typed
    `model_name`/`residency` attributes) because `CLAUDE.md` and the tests still name it, not
    because the lookup itself lives here -- that lookup is `alias_for`
    (`app.residency.ResidencyAllowList`), which `validate_model_for_residency` below calls.

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


def validate_model_for_residency(
    model_name: str, residency: str, *, settings: Settings | None = None
) -> str:
    """Returns the bare (gateway alias) model name if `model_name` is allow-listed for
    `residency`; raises `ModelNotAllowedForResidency` otherwise. A thin call to
    `settings.residency_allow_list.alias_for` (`app.residency.ResidencyAllowList`, spec A4 /
    #111) -- this module holds no lookup of its own, only the ADR-0009-specific exception name
    and attributes. Synchronous and side-effect-free beyond reading `settings.residency_allow_list`
    (`get_settings()` when not given) -- no client, no network call, no database.
    """
    s = settings or get_settings()
    allow_list = s.residency_allow_list
    assert allow_list is not None  # set by Settings construction
    try:
        return allow_list.alias_for(residency, model_name)
    except ResidencyUnresolved as exc:
        raise ModelNotAllowedForResidency(model_name, residency) from exc


# Cache of per-tenant gateway connections (ADR-0009: "any cache of them is keyed by tenant"), so
# two tenants never share a connection or credential, and resolving the same tenant twice reuses
# one connection rather than reconnecting. The tenant's credential -- not its momentary model
# choice -- is the resource worth not duplicating, so the provider (the connection) is keyed by
# tenant id alone; the model objects built on it are keyed by (tenant id, model name), so a
# tenant that changes its `model` setting (#105) gets the new model on its same connection on
# the very next resolution, never the cached old one.
_tenant_providers: dict[UUID, OpenAIProvider] = {}
_tenant_chat_models: dict[tuple[UUID, str], OpenAIChatModel] = {}


def build_tenant_chat_model(
    tenant_id: UUID,
    bare_model_name: str,
    credential: SecretStr,
    *,
    settings: Settings | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> OpenAIChatModel:
    """Builds (or reuses) the chat model client for `tenant_id` and `bare_model_name`, from that
    tenant's own gateway credential -- never a process-wide key. The tenant's connection is
    cached per tenant id: two calls for the same tenant share one client whatever the model
    name, the same name twice returns the exact same model object, and a different `tenant_id`
    always gets a distinct client. Carries an explicit wall-clock deadline
    (`ModelSettings.timeout` = `Settings.llm_call_timeout_seconds`), replacing the client
    library's multi-minute default.

    `http_client` is test-only (an already-configured `httpx.AsyncClient`, e.g. one backed by a
    `MockTransport`); production call sites never pass it.
    """
    cached = _tenant_chat_models.get((tenant_id, bare_model_name))
    if cached is not None:
        return cached
    s = settings or get_settings()
    if not s.litellm_base_url:
        raise RuntimeError(
            "LITELLM_BASE_URL must be configured to build a per-tenant model client: a gateway "
            "credential is only valid against the gateway, never a raw provider endpoint."
        )
    provider = _tenant_providers.get(tenant_id)
    if provider is None:
        provider = OpenAIProvider(
            base_url=s.litellm_base_url,
            api_key=credential.get_secret_value(),
            http_client=http_client,
        )
        _tenant_providers[tenant_id] = provider
    model = OpenAIChatModel(
        bare_model_name,
        provider=provider,
        settings=ModelSettings(timeout=s.llm_call_timeout_seconds),
    )
    _tenant_chat_models[(tenant_id, bare_model_name)] = model
    return model


def reset_tenant_chat_model_cache() -> None:
    """Test-only: clears the per-tenant chat model cache between test cases."""
    _tenant_chat_models.clear()
    _tenant_providers.clear()


def resolve_tenant_chat_model(
    record: TenantRecord, *, settings: Settings | None = None
) -> OpenAIChatModel:
    """The per-tenant model-resolution entry point (ADR-0009, Spec 7 / #54) -- a function of the
    tenant record and settings, with no database read of its own (#105).

    In order: the record's residency and its allow-listed route (`ResidencyUnresolved` if none
    is recorded or the allow-list does not know it -- ADR-0008 fails closed, no default); the
    model name -- the tenant's own `record.settings.model`, else the deployment default
    `Settings.llm_model` -- validated against that residency's allow-list, raising
    `ModelNotAllowedForResidency` before anything else happens (a rejected model never reaches
    credential resolution); then the tenant's own gateway credential (the file
    `record.gateway_credential_alias` names, `GatewayCredentialUnavailable` if either is missing)
    and its cached client.
    """
    s = settings or get_settings()
    residency, _route = tenant_residency_route(record, settings=s)
    name = record.settings.model or s.llm_model
    bare = validate_model_for_residency(name, residency, settings=s)
    credential = resolve_gateway_credential(record, settings=s)
    return build_tenant_chat_model(record.tenant_id, bare, credential, settings=s)


__all__ = [
    "ModelNotAllowedForResidency",
    "build_tenant_chat_model",
    "reset_tenant_chat_model_cache",
    "resolve_tenant_chat_model",
    "validate_model_for_residency",
]
