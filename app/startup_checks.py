"""Fail-closed residency and model-allow-list startup checks (ADR-0008, Spec 8 / #59).

`run_startup_checks` is the one function both the HTTP API's construction/lifespan path
(`app.main.create_app` / `app.main.lifespan`) and the MCP server's own startup path
(`app.mcp.server.main`) call before either process ever accepts a request or a tool call. It
walks every content-bearing endpoint this deployment is actually configured to reach --
the model/gateway host (`Settings.litellm_base_url`), the embedding endpoint (the same host in
gateway mode, otherwise the embedding client's own default), and the trace sink
(`Settings.langfuse_host`) -- and checks each one against the allow-list for this deployment's
own residency (`RESIDENCY_ALLOW_LIST[settings.residency]`, app/config.py). It also runs the model
allow-list check a tenant's chosen model is validated against at request time
(`app.llm.validate_model_for_residency`, S7-T4 / #54) against the deployment's own default model,
in this same pass -- an operator fixing a residency mismatch and a model mismatch is one error to
read, not two validations to reconcile by hand.

Every failure is a `ResidencyConfigurationError` naming both the offending residency and the
offending endpoint, so an operator can fix the deployment's configuration without reading source.
This never touches a tenant's own `control.tenants.residency` (that is resolved per request by
`app.residency.resolve_residency_route`, fails closed there with `ResidencyUnresolved`) -- this
module only checks what the deployment itself is configured to reach.
"""

from __future__ import annotations

import fnmatch
from urllib.parse import urlparse

from app.config import RESIDENCY_ALLOW_LIST, Settings
from app.llm import ModelNotAllowedForResidency, validate_model_for_residency

# The embedding client's own default endpoint when no gateway is configured
# (`openai.AsyncOpenAI()` with no `base_url`, see app/embeddings.py's `_client`).
_DEFAULT_OPENAI_EMBEDDING_ENDPOINT = "https://api.openai.com/v1"


class ResidencyConfigurationError(RuntimeError):
    """A configured endpoint sits outside its residency's allow-list, or the deployment's
    residency has no allow-list entry at all. The message always names both the offending
    residency and the offending endpoint -- an operator must be able to fix this without reading
    source."""


def _host(endpoint: str) -> str:
    """The bare hostname of a URL or a bare host string alike (`langfuse_host` is configured
    without a scheme; `litellm_base_url` and the allow-list's `embedding_endpoint` are full
    URLs)."""
    parsed = urlparse(endpoint if "://" in endpoint else f"//{endpoint}")
    return (parsed.netloc or endpoint).split(":", 1)[0]


def _host_matches_any(host: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(host, pattern) for pattern in patterns)


def run_startup_checks(settings: Settings) -> None:
    """Raises `ResidencyConfigurationError` before the process may accept its first request or
    tool call when:

    - this deployment's own residency (`settings.residency`) has no entry in
      `RESIDENCY_ALLOW_LIST` (defensive -- `Settings` construction already rejects an unknown
      residency, but this function never trusts that as its only guard);
    - a configured model/gateway host (`settings.litellm_base_url`) is not on that residency's
      allow-listed model host patterns;
    - the deployment's default model (`settings.llm_model`) is not on that residency's model
      allow-list (`RESIDENCY_MODEL_ALLOW_LIST`, checked only when a gateway is configured -- a
      bare gateway alias is only ever meaningful against the gateway's own `model_list`, never
      against a direct-provider `<provider>:<model>` id);
    - the embedding endpoint this deployment will actually reach (the gateway host in gateway
      mode, otherwise the embedding client's own default) is not that residency's allow-listed
      embedding endpoint, and not otherwise covered by its model host patterns;
    - a configured trace sink (`settings.langfuse_host`) is not that residency's allow-listed
      trace sink host.

    A fully valid, allow-listed configuration (including the common case of no gateway and no
    tracing configured at all) returns without raising.
    """
    residency = settings.residency
    route = RESIDENCY_ALLOW_LIST.get(residency)
    if route is None:
        raise ResidencyConfigurationError(
            f"residency {residency!r} has no entry in RESIDENCY_ALLOW_LIST "
            f"(configured residencies: {sorted(RESIDENCY_ALLOW_LIST)})"
        )

    if settings.litellm_base_url:
        gateway_host = _host(settings.litellm_base_url)
        if not _host_matches_any(gateway_host, route.model_host_patterns):
            raise ResidencyConfigurationError(
                f"residency {residency!r}: model/gateway host {gateway_host!r} "
                f"(LITELLM_BASE_URL={settings.litellm_base_url!r}) is not on its allow-listed "
                f"host patterns {route.model_host_patterns}"
            )
        # Model allow-list check (S7-T4 / #54), run in this same pass: a bare gateway alias is
        # only checkable once we know a gateway is actually configured.
        try:
            validate_model_for_residency(settings.llm_model, residency)
        except ModelNotAllowedForResidency as exc:
            raise ResidencyConfigurationError(
                f"residency {residency!r}: default model {settings.llm_model!r} is not on its "
                f"model allow-list -- {exc}"
            ) from exc
        embedding_host = gateway_host
    else:
        embedding_host = _host(_DEFAULT_OPENAI_EMBEDDING_ENDPOINT)

    if embedding_host != _host(route.embedding_endpoint) and not _host_matches_any(
        embedding_host, route.model_host_patterns
    ):
        raise ResidencyConfigurationError(
            f"residency {residency!r}: embedding endpoint host {embedding_host!r} is not its "
            f"allow-listed embedding endpoint {route.embedding_endpoint!r}"
        )

    if settings.langfuse_host:
        trace_host = _host(settings.langfuse_host)
        if trace_host != route.trace_sink_host:
            raise ResidencyConfigurationError(
                f"residency {residency!r}: trace sink host {trace_host!r} "
                f"(LANGFUSE_HOST={settings.langfuse_host!r}) is not its allow-listed trace sink "
                f"{route.trace_sink_host!r}"
            )


__all__ = ["ResidencyConfigurationError", "run_startup_checks"]
