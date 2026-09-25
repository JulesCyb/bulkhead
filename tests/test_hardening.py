"""Guard tests: SSE framing, the dev-headers environment guard, and residency/embedding
configuration-property tests (ADR-0008, spec 8 / issue #58)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.agents import _sse
from app.config import RESIDENCY_ALLOW_LIST, Settings
from app.main import check_auth_mode


def test_sse_framing_preserves_newlines():
    assert _sse("hello") == "data: hello\n\n"
    # A delta containing newlines must become multiple data: lines (the client
    # reassembles them), never a raw line without the data: prefix.
    assert _sse("line one\nline two") == "data: line one\ndata: line two\n\n"
    assert _sse("") == "data: \n\n"


def test_dev_headers_refused_outside_dev():
    with pytest.raises(RuntimeError):
        check_auth_mode(Settings(environment="prod", auth_mode="dev-headers"))
    # No raise: dev-headers in dev/test, jwt anywhere.
    check_auth_mode(Settings(environment="dev", auth_mode="dev-headers"))
    check_auth_mode(Settings(environment="test", auth_mode="dev-headers"))
    check_auth_mode(Settings(environment="prod", auth_mode="jwt"))


# --- Residency allow-list and required embedding configuration (issue #58) ---


def test_settings_requires_embedding_provider():
    """No default embedding provider: unset must fail construction, not reach a default
    endpoint. Verified directly against Settings, no network call."""
    with pytest.raises(ValidationError, match="EMBEDDING_PROVIDER"):
        Settings(embedding_provider=None, embedding_model="text-embedding-3-small")


def test_settings_requires_embedding_model():
    with pytest.raises(ValidationError, match="EMBEDDING_MODEL"):
        Settings(embedding_provider="openai", embedding_model=None)


def test_settings_residency_allow_list_has_at_least_two_residencies():
    # Data-driven, not hard-coded per provider: at least two residencies for tests, each with
    # allowed model/gateway host patterns, an allowed embedding endpoint, and a trace sink host.
    assert len(RESIDENCY_ALLOW_LIST) >= 2
    for route in RESIDENCY_ALLOW_LIST.values():
        assert route.model_host_patterns
        assert route.embedding_endpoint
        assert route.trace_sink_host


def test_settings_rejects_unknown_residency():
    with pytest.raises(ValidationError, match="Unknown residency"):
        Settings(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            residency="mars",
        )


def test_settings_valid_configuration_constructs_cleanly():
    settings = Settings(
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        residency="eu",
    )
    assert settings.residency == "eu"
    assert settings.residency_route == RESIDENCY_ALLOW_LIST["eu"]

    # A second, distinct residency also constructs cleanly and resolves its own route.
    other = Settings(
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        residency="us",
    )
    assert other.residency_route != settings.residency_route
