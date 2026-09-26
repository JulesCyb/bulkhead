"""Fail-closed residency and model-allow-list startup checks (ADR-0008, spec A4 / #94, #110).

`run_startup_checks` is the one function both the HTTP API's construction/lifespan path
(`app.main.create_app` / `app.main.lifespan`) and the MCP server's own startup path
(`app.mcp.server.main`) call before either process ever accepts a request or a tool call. It is a
thin call to `settings.residency_allow_list.check_deployment(settings)`
(`app.residency.ResidencyAllowList`) -- that method walks every content-bearing endpoint this
deployment is actually configured to reach (the model/gateway host, the embedding endpoint, the
deployment's default model, and the trace sink) against the allow-list for this deployment's own
residency, in one pass; see its docstring for the exact checks.

`ResidencyConfigurationError` is kept here as a subclass of `app.residency.ResidencyUnresolved`
(itself raised by `check_deployment`) purely for callers/tests that still name this module's own
type -- unifying the two names is ticket #111's job. Every failure names both the offending
residency and the offending endpoint, so an operator can fix the deployment's configuration
without reading source.

This never touches a tenant's own `control.tenants.residency` (that is resolved per request by
`app.residency.resolve_residency_route`, fails closed there with `ResidencyUnresolved`) -- this
module only checks what the deployment itself is configured to reach.
"""

from __future__ import annotations

from app.config import Settings
from app.residency import ResidencyUnresolved


class ResidencyConfigurationError(ResidencyUnresolved):
    """A configured endpoint sits outside its residency's allow-list, or the deployment's
    residency has no allow-list entry at all. The message always names both the offending
    residency and the offending endpoint -- an operator must be able to fix this without reading
    source."""


def run_startup_checks(settings: Settings) -> None:
    """Raises `ResidencyConfigurationError` before the process may accept its first request or
    tool call when `settings.residency_allow_list.check_deployment(settings)` finds this
    deployment's own configuration outside its residency's allow-list -- see that method's
    docstring for the exact checks. A fully valid, allow-listed configuration returns without
    raising.
    """
    assert settings.residency_allow_list is not None  # set by Settings construction
    try:
        settings.residency_allow_list.check_deployment(settings)
    except ResidencyUnresolved as exc:
        raise ResidencyConfigurationError(str(exc)) from exc


__all__ = ["ResidencyConfigurationError", "run_startup_checks"]
