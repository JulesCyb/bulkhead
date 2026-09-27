"""Fail-closed residency and model-allow-list startup checks (ADR-0008, Spec 8 / #59, the
mandatory-gateway follow-up ai-app-starter#7 / ADR-0009, and spec A4 / #94, #110).

Configuration-property tests only: no network call, no database. Mirrors
tests/test_hardening.py's existing residency/embedding configuration tests and the
construction-time/lifespan pattern used there for issue #15. `run_startup_checks` is now a thin
call to `Settings.residency_allow_list.check_deployment` (`app.residency.ResidencyAllowList`) --
these tests exercise it through `Settings`/`run_startup_checks`, never by patching a module-level
allow-list.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.residency import ResidencyAllowList
from app.startup_checks import ResidencyConfigurationError, run_startup_checks

_EMBEDDING_KWARGS = {
    "embedding_provider": "openai",
    "embedding_model": "text-embedding-3-small",
}
# Includes the compose default gateway host and a model that is actually allow-listed for it
# ("litellm" -> residency "eu" -> ["claude-eu", "embeddings"], config/residency.toml) so tests
# that don't care about a specific `litellm_base_url`/`llm_model` value don't each have to supply
# one -- `Settings()` now refuses to construct without a gateway URL at all (ADR-0009), and with
# one configured, `run_startup_checks` always validates the default model against it too.
_VALID_KWARGS = {
    **_EMBEDDING_KWARGS,
    "litellm_base_url": "http://litellm:4000",
    "llm_model": "claude-eu",
}


def test_missing_gateway_url_fails_settings_construction_in_every_environment():
    """ai-app-starter#7 review finding (ADR-0009): the gateway is mandatory, not an optional
    profile -- `Settings()` must refuse to construct at all with no `LITELLM_BASE_URL`, before
    `run_startup_checks` (or anything else) ever runs. Not conditioned on `environment` or
    `auth_mode`: every deployment's compose stack runs the gateway."""
    with pytest.raises(ValidationError, match="LITELLM_BASE_URL"):
        Settings(
            residency="eu",
            litellm_base_url=None,
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )


def test_empty_gateway_url_also_fails_settings_construction():
    """An explicitly empty string is treated the same as unset -- not a valid "no gateway" value
    (ADR-0009)."""
    with pytest.raises(ValidationError, match="LITELLM_BASE_URL"):
        Settings(
            residency="eu",
            litellm_base_url="",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )


def test_fully_valid_configuration_with_compose_default_gateway_starts_cleanly_no_tracing():
    """The common case: the compose stack's own default gateway host ("litellm", allow-listed
    under residency "eu"), no tracing configured. Must not raise."""
    settings = Settings(residency="eu", **_VALID_KWARGS)
    run_startup_checks(settings)  # no raise


def test_fully_valid_configuration_with_gateway_and_matching_model_starts_cleanly():
    settings = Settings(
        residency="eu",
        litellm_base_url="https://gateway.eu.litellm.internal",
        llm_model="claude-eu",
        **_EMBEDDING_KWARGS,
    )
    run_startup_checks(settings)  # no raise


def test_fully_valid_configuration_with_matching_trace_sink_starts_cleanly():
    settings = Settings(residency="eu", langfuse_host="eu.cloud.langfuse.com", **_VALID_KWARGS)
    run_startup_checks(settings)  # no raise


def test_mismatched_gateway_host_raises_naming_residency_and_endpoint():
    """A gateway host outside the deployment's own residency's allow-listed host patterns must
    raise before the application ever accepts a request."""
    settings = Settings(
        residency="eu",
        litellm_base_url="https://gateway.us.litellm.internal",
        **_EMBEDDING_KWARGS,
    )
    with pytest.raises(ResidencyConfigurationError) as exc_info:
        run_startup_checks(settings)
    message = str(exc_info.value)
    assert "eu" in message
    assert "gateway.us.litellm.internal" in message


def test_us_gateway_host_is_rejected_under_eu_residency():
    """The original finding (#63): a residency's allow-list must be able to reject the *other*
    residency's own gateway host, not just an unrelated made-up one."""
    us_route = Settings(residency="us", **_VALID_KWARGS).residency_route
    settings = Settings(
        residency="eu",
        litellm_base_url=f"https://{us_route.model_host_patterns[0].lstrip('*.')}",
        **_EMBEDDING_KWARGS,
    )
    with pytest.raises(ResidencyConfigurationError, match="eu"):
        run_startup_checks(settings)


def test_eu_gateway_host_is_rejected_under_us_residency():
    eu_route = Settings(residency="eu", **_VALID_KWARGS).residency_route
    # "litellm" (the compose default eu host) is bare, not a glob -- use one of the eu-specific
    # glob hosts instead so this exercises a real hostname, not the pattern itself.
    eu_host_pattern = next(p for p in eu_route.model_host_patterns if p.startswith("*."))
    settings = Settings(
        residency="us",
        litellm_base_url=f"https://{eu_host_pattern.lstrip('*.')}",
        **_EMBEDDING_KWARGS,
    )
    with pytest.raises(ResidencyConfigurationError, match="us"):
        run_startup_checks(settings)


def test_a_global_provider_host_is_rejected_under_eu_residency():
    """A host reachable from every jurisdiction (an Anthropic/OpenAI direct endpoint) must never
    pass a residency's allow-list check -- this is exactly the original finding: `eu`'s
    model_host_patterns must not also cover a global provider domain."""
    settings = Settings(
        residency="eu", litellm_base_url="https://api.anthropic.com", **_EMBEDDING_KWARGS
    )
    with pytest.raises(ResidencyConfigurationError, match="eu"):
        run_startup_checks(settings)


def test_a_global_provider_host_is_rejected_under_us_residency():
    settings = Settings(
        residency="us", litellm_base_url="https://api.openai.com", **_EMBEDDING_KWARGS
    )
    with pytest.raises(ResidencyConfigurationError, match="us"):
        run_startup_checks(settings)


def test_mismatched_trace_sink_raises_naming_residency_and_endpoint():
    settings = Settings(residency="eu", langfuse_host="us.cloud.langfuse.com", **_VALID_KWARGS)
    with pytest.raises(ResidencyConfigurationError) as exc_info:
        run_startup_checks(settings)
    message = str(exc_info.value)
    assert "eu" in message
    assert "us.cloud.langfuse.com" in message


def test_model_not_on_residency_allow_list_raises_in_the_same_pass():
    """The model allow-list check (S7-T4 / #54) runs together with the residency checks in this
    one startup pass, not as a second validation to reconcile separately."""
    settings = Settings(
        residency="eu",
        litellm_base_url="https://gateway.eu.litellm.internal",
        llm_model="mistral-large",
        **_EMBEDDING_KWARGS,
    )
    with pytest.raises(ResidencyConfigurationError) as exc_info:
        run_startup_checks(settings)
    message = str(exc_info.value)
    assert "eu" in message
    assert "mistral-large" in message


def test_model_allowed_in_us_but_not_eu_is_residency_specific():
    """`claude` (bare) is on the US allow-list but not the EU one -- proves the check is against
    the deployment's own residency, not any residency."""
    settings = Settings(
        residency="eu",
        litellm_base_url="https://gateway.eu.litellm.internal",
        llm_model="claude",
        **_EMBEDDING_KWARGS,
    )
    with pytest.raises(ResidencyConfigurationError):
        run_startup_checks(settings)

    us_settings = Settings(
        residency="us",
        litellm_base_url="https://gateway.us.litellm.internal",
        llm_model="claude",
        **_EMBEDDING_KWARGS,
    )
    run_startup_checks(us_settings)  # no raise


def test_unconfigured_residency_raises_naming_the_residency():
    """Defensive: even though Settings construction already rejects an unknown residency against
    the very same allow-list object, `check_deployment` never trusts that as its only guard -- an
    allow-list swapped out for one missing the deployment's own residency after construction
    still fails closed here. No global to patch: the allow-list is swapped on the settings object
    itself, the seam spec A4 (#110) exists for."""
    settings = Settings(residency="eu", **_VALID_KWARGS)
    settings.residency_allow_list = ResidencyAllowList.from_data(
        {
            "residency": {
                "us": {
                    "model_host_patterns": ["gateway-us.internal"],
                    "embedding_endpoint": "https://gateway-us.internal/v1",
                    "trace_sink_host": "us.cloud.langfuse.com",
                }
            }
        }
    )
    with pytest.raises(ResidencyConfigurationError) as exc_info:
        run_startup_checks(settings)
    assert "eu" in str(exc_info.value)


# --- Wired into the ASGI application's construction/lifespan path (issue #59 / ADR-0008) ---


def test_create_app_raises_immediately_for_mismatched_residency_configuration():
    """Same construction-time enforcement as check_auth_mode (issue #15): building the
    application object with a deliberately mismatched residency configuration raises
    immediately, with no ASGI lifespan event ever firing."""
    from app.main import create_app

    mismatched_settings = Settings(
        environment="dev",
        auth_mode="dev-headers",
        residency="eu",
        langfuse_host="us.cloud.langfuse.com",
        **_VALID_KWARGS,
    )
    with pytest.raises(ResidencyConfigurationError):
        create_app(settings=mismatched_settings)


async def test_lifespan_also_raises_for_mismatched_residency_configuration(monkeypatch):
    """Construction succeeds with a valid configuration. Driving the ASGI lifespan directly
    afterwards, with `get_settings()` now returning a mismatched residency configuration, must
    still raise -- a second, independent execution of the guard, not a value cached from
    construction."""
    from app import main as main_module

    good_settings = Settings(environment="dev", auth_mode="dev-headers", **_VALID_KWARGS)
    app_obj = main_module.create_app(settings=good_settings)

    mismatched_settings = Settings(
        environment="dev",
        auth_mode="dev-headers",
        residency="eu",
        langfuse_host="us.cloud.langfuse.com",
        **_VALID_KWARGS,
    )
    monkeypatch.setattr(main_module, "get_settings", lambda: mismatched_settings)

    with pytest.raises(ResidencyConfigurationError):
        async with app_obj.router.lifespan_context(app_obj):
            pass


# --- Wired into the MCP server's own startup path (issue #59 / ADR-0008) ---


def test_mcp_server_main_runs_startup_checks_before_serving(monkeypatch):
    """A valid configuration: the MCP server's entry point runs the same startup checks and then
    starts the transport.

    `run_role_rls_guard` (issue #81) is substituted with a no-op fake: this module's own tests
    are configuration-property tests only (module docstring above) -- no network call, no
    database -- and the guard itself has its own dedicated tests
    (`tests/test_mcp_context.py::test_main_runs_the_role_rls_guard_before_serving` and
    `::test_main_refuses_to_start_when_the_role_rls_guard_fails`)."""
    from app.mcp import server as mcp_server

    good_settings = Settings(residency="eu", **_VALID_KWARGS)
    monkeypatch.setattr(mcp_server, "get_settings", lambda: good_settings)

    async def _passing_guard() -> None:
        return None

    monkeypatch.setattr(mcp_server, "run_role_rls_guard", _passing_guard)

    calls: list[str] = []
    monkeypatch.setattr(
        mcp_server.server, "run", lambda transport=None: calls.append(transport or "")
    )

    mcp_server.main()

    assert calls == ["stdio"]


def test_mcp_server_main_raises_and_never_serves_on_mismatched_configuration(monkeypatch):
    """A deliberately mismatched configuration must raise before the MCP server ever starts
    accepting a tool call on its transport."""
    from app.mcp import server as mcp_server

    mismatched_settings = Settings(
        residency="eu", langfuse_host="us.cloud.langfuse.com", **_VALID_KWARGS
    )
    monkeypatch.setattr(mcp_server, "get_settings", lambda: mismatched_settings)

    calls: list[str] = []
    monkeypatch.setattr(
        mcp_server.server, "run", lambda transport=None: calls.append(transport or "")
    )

    with pytest.raises(ResidencyConfigurationError):
        mcp_server.main()

    assert calls == []
