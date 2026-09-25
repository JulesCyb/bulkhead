"""Fail-closed residency and model-allow-list startup checks (ADR-0008, Spec 8 / #59, #63).

`run_startup_checks` is the one function both the HTTP API's construction/lifespan path
(`app.main.create_app` / `app.main.lifespan`) and the MCP server's own startup path
(`app.mcp.server.main`) call before either process ever accepts a request or a tool call. It
walks every content-bearing endpoint this deployment is actually configured to reach --
the model/gateway host (`Settings.litellm_base_url`), the embedding endpoint (the same gateway
host -- embeddings are only ever reachable through it, `app/embeddings.py`), and the trace sink
(`Settings.langfuse_host`) -- and checks each one against the allow-list for this deployment's
own residency (`RESIDENCY_ALLOW_LIST[settings.residency]`, loaded by `app.config` from
`config/residency.toml`). It also runs the model allow-list check a tenant's chosen model is
validated against at request time (`app.llm.validate_model_for_residency`, S7-T4 / #54) against
the deployment's own default model, in this same pass -- an operator fixing a residency mismatch
and a model mismatch is one error to read, not two validations to reconcile by hand.

Every failure is a `ResidencyConfigurationError` naming both the offending residency and the
offending endpoint, so an operator can fix the deployment's configuration without reading source.
This never touches a tenant's own `control.tenants.residency` (that is resolved per request by
`app.residency.resolve_residency_route`, fails closed there with `ResidencyUnresolved`) -- this
module only checks what the deployment itself is configured to reach.

The gateway/model/embedding checks below are unconditional, not "run only when a gateway is
configured" (ai-app-starter#7 review finding, ADR-0009): `Settings` itself now refuses to
construct with an unset or empty `LITELLM_BASE_URL`
(`app.config.Settings._require_gateway_configured`) -- every deployment's compose stack always
runs its own LiteLLM gateway, so by the time a `Settings` object reaches this function,
`settings.litellm_base_url` is guaranteed truthy. This function does not re-check that it is
set (that would just duplicate the `Settings` validator); it checks that the gateway host that
*is* configured sits inside this deployment's residency allow-list, which is the one thing
`Settings` construction cannot know on its own.

Deliberately no fallback to a hard-coded "default" embedding/model host (there used to be one,
`https://api.openai.com/v1`, checked when no gateway was configured -- removed in #63, and made
moot entirely once #7 made the gateway mandatory at `Settings` construction): a provider's global
endpoint is reachable from every jurisdiction, so allowing it under every residency's allow-list
made this check unable to ever reject a genuinely out-of-residency host.
"""

from __future__ import annotations

import fnmatch
from urllib.parse import urlparse

from app.config import RESIDENCY_ALLOW_LIST, Settings
from app.llm import ModelNotAllowedForResidency, validate_model_for_residency


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
    - the configured model/gateway host (`settings.litellm_base_url`) is not on that residency's
      allow-listed model host patterns -- checked unconditionally: `Settings` construction itself
      already refuses to leave `litellm_base_url` unset (ADR-0009,
      `app.config.Settings._require_gateway_configured`), so there is no "no gateway configured"
      branch here to skip;
    - the deployment's default model (`settings.llm_model`) is not on that residency's model
      allow-list (`RESIDENCY_MODEL_ALLOW_LIST`) -- a bare gateway alias is only ever meaningful
      against the gateway's own `model_list`;
    - the embedding endpoint the deployment will actually reach (the same gateway host, the only
      path `app.embeddings` ever calls) is not that residency's allow-listed embedding endpoint,
      and not otherwise covered by its model host patterns;
    - a configured trace sink (`settings.langfuse_host`) is not that residency's allow-listed
      trace sink host.

    A fully valid, allow-listed configuration returns without raising.
    """
    residency = settings.residency
    route = RESIDENCY_ALLOW_LIST.get(residency)
    if route is None:
        raise ResidencyConfigurationError(
            f"residency {residency!r} has no entry in RESIDENCY_ALLOW_LIST "
            f"(configured residencies: {sorted(RESIDENCY_ALLOW_LIST)})"
        )

    # Unconditional (ai-app-starter#7 review finding, ADR-0009): `settings.litellm_base_url` is
    # guaranteed truthy here -- `Settings` refuses to construct without it -- so there is no
    # "gateway not configured" branch to skip past.
    gateway_host = _host(settings.litellm_base_url)  # type: ignore[arg-type]
    if not _host_matches_any(gateway_host, route.model_host_patterns):
        raise ResidencyConfigurationError(
            f"residency {residency!r}: model/gateway host {gateway_host!r} "
            f"(LITELLM_BASE_URL={settings.litellm_base_url!r}) is not on its allow-listed "
            f"host patterns {route.model_host_patterns}"
        )
    # Model allow-list check (S7-T4 / #54), run in this same pass.
    try:
        validate_model_for_residency(settings.llm_model, residency)
    except ModelNotAllowedForResidency as exc:
        raise ResidencyConfigurationError(
            f"residency {residency!r}: default model {settings.llm_model!r} is not on its "
            f"model allow-list -- {exc}"
        ) from exc

    # Embeddings are only ever reachable through this same gateway host (ADR-0009,
    # app/embeddings.py refuses to build a client with no gateway configured).
    embedding_host = gateway_host
    if embedding_host != _host(route.embedding_endpoint) and not _host_matches_any(
        embedding_host, route.model_host_patterns
    ):
        raise ResidencyConfigurationError(
            f"residency {residency!r}: embedding endpoint host {embedding_host!r} is not "
            f"its allow-listed embedding endpoint {route.embedding_endpoint!r}"
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
