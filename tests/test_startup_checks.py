"""Fail-closed residency and model-allow-list startup checks (ADR-0008, Spec 8 / #59).

Configuration-property tests only: no network call, no database. Mirrors
tests/test_hardening.py's existing residency/embedding configuration tests and the
construction-time/lifespan pattern used there for issue #15.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.startup_checks import ResidencyConfigurationError, run_startup_checks

_VALID_KWARGS = {"embedding_provider": "openai", "embedding_model": "text-embedding-3-small"}


def test_fully_valid_configuration_starts_cleanly_no_gateway_no_tracing():
    """The common case: no gateway, no tracing configured, deployment residency's default model
    used directly against the provider. Must not raise."""
    settings = Settings(residency="eu", **_VALID_KWARGS)
    run_startup_checks(settings)  # no raise


def test_fully_valid_configuration_with_gateway_and_matching_model_starts_cleanly():
    settings = Settings(
        residency="eu",
        litellm_base_url="https://gateway.eu.litellm.internal",
        llm_model="claude-eu",
        **_VALID_KWARGS,
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
        **_VALID_KWARGS,
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
        **_VALID_KWARGS,
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
        **_VALID_KWARGS,
    )
    with pytest.raises(ResidencyConfigurationError, match="us"):
        run_startup_checks(settings)


def test_a_global_provider_host_is_rejected_under_eu_residency():
    """A host reachable from every jurisdiction (an Anthropic/OpenAI direct endpoint) must never
    pass a residency's allow-list check -- this is exactly the original finding: `eu`'s
    model_host_patterns must not also cover a global provider domain."""
    settings = Settings(
        residency="eu", litellm_base_url="https://api.anthropic.com", **_VALID_KWARGS
    )
    with pytest.raises(ResidencyConfigurationError, match="eu"):
        run_startup_checks(settings)


def test_a_global_provider_host_is_rejected_under_us_residency():
    settings = Settings(residency="us", litellm_base_url="https://api.openai.com", **_VALID_KWARGS)
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
        **_VALID_KWARGS,
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
        **_VALID_KWARGS,
    )
    with pytest.raises(ResidencyConfigurationError):
        run_startup_checks(settings)

    us_settings = Settings(
        residency="us",
        litellm_base_url="https://gateway.us.litellm.internal",
        llm_model="claude",
        **_VALID_KWARGS,
    )
    run_startup_checks(us_settings)  # no raise


def test_unconfigured_residency_raises_naming_the_residency(monkeypatch):
    """Defensive: even though Settings construction already rejects an unknown residency, this
    function never trusts that as its only guard -- an allow-list entry removed after
    construction (or a residency injected another way) still fails closed here."""
    import app.startup_checks as startup_checks_module

    settings = Settings(residency="eu", **_VALID_KWARGS)
    monkeypatch.setattr(startup_checks_module, "RESIDENCY_ALLOW_LIST", {})
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
    starts the transport."""
    from app.mcp import server as mcp_server

    good_settings = Settings(residency="eu", **_VALID_KWARGS)
    monkeypatch.setattr(mcp_server, "get_settings", lambda: good_settings)

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
