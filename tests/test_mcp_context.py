"""The MCP server's development-only context provider (identity_id naming, env-based) and its
startup transport guard (issue #48 / ADR-0005)."""

from __future__ import annotations

import uuid

import pytest

from app.config import Settings
from app.mcp import server as mcp_server

_VALID_KWARGS = {"embedding_provider": "openai", "embedding_model": "text-embedding-3-small"}


def test_context_from_env_uses_renamed_identity_setting(monkeypatch):
    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    monkeypatch.setattr(
        mcp_server,
        "get_settings",
        lambda: Settings(mcp_tenant_id=str(tenant_id), mcp_identity_id=str(identity_id)),
    )
    ctx = mcp_server._context_from_env()
    assert ctx.tenant_id == tenant_id
    assert ctx.identity_id == identity_id


def test_retired_mcp_user_id_setting_is_not_silently_accepted(monkeypatch):
    # The retired env var name must not populate the renamed setting.
    monkeypatch.delenv("MCP_IDENTITY_ID", raising=False)
    monkeypatch.setenv("MCP_TENANT_ID", str(uuid.uuid4()))
    monkeypatch.setenv("MCP_USER_ID", str(uuid.uuid4()))
    settings = Settings(_env_file=None)
    assert settings.mcp_identity_id is None
    monkeypatch.setattr(mcp_server, "get_settings", lambda: settings)
    with pytest.raises(RuntimeError):
        mcp_server._context_from_env()


# --- MCP transport setting (issue #48 / ADR-0005) ---


def test_mcp_transport_defaults_to_stdio():
    settings = Settings(_env_file=None, environment="dev", auth_mode="dev-headers", **_VALID_KWARGS)
    assert settings.mcp_transport == "stdio"


def test_mcp_transport_accepts_each_explicit_value():
    for value in ("stdio", "streamable-http"):
        settings = Settings(
            _env_file=None,
            environment="dev",
            auth_mode="dev-headers",
            mcp_transport=value,
            **_VALID_KWARGS,
        )
        assert settings.mcp_transport == value


def test_mcp_transport_rejects_unknown_value():
    with pytest.raises(ValueError):
        Settings(
            _env_file=None,
            environment="dev",
            auth_mode="dev-headers",
            mcp_transport="sse",
            **_VALID_KWARGS,
        )


# --- check_mcp_mode: the startup transport guard (issue #48 / ADR-0005) ---


def test_check_mcp_mode_raises_for_stdio_outside_dev_and_test():
    with pytest.raises(RuntimeError, match="stdio"):
        mcp_server.check_mcp_mode(
            Settings(
                _env_file=None,
                environment="prod",
                auth_mode="jwt",
                mcp_transport="stdio",
                **_VALID_KWARGS,
            )
        )


def test_check_mcp_mode_raises_for_streamable_http_without_a_configured_verifier():
    # Regardless of environment -- even dev/test must not silently accept a networked transport
    # with nothing to check connections against.
    for environment in ("dev", "test", "prod"):
        with pytest.raises(RuntimeError, match="streamable-http"):
            mcp_server.check_mcp_mode(
                Settings(
                    _env_file=None,
                    environment=environment,
                    auth_mode="jwt",
                    mcp_transport="streamable-http",
                    jwt_verification_key=None,
                    **_VALID_KWARGS,
                )
            )


def test_check_mcp_mode_does_not_raise_for_stdio_in_dev_or_test():
    for environment in ("dev", "test"):
        mcp_server.check_mcp_mode(
            Settings(
                _env_file=None,
                environment=environment,
                auth_mode="dev-headers",
                mcp_transport="stdio",
                **_VALID_KWARGS,
            )
        )  # no raise


def test_check_mcp_mode_does_not_raise_for_streamable_http_with_a_verifier_configured():
    for environment in ("dev", "test", "prod"):
        mcp_server.check_mcp_mode(
            Settings(
                _env_file=None,
                environment=environment,
                auth_mode="jwt",
                mcp_transport="streamable-http",
                jwt_verification_key="a-verification-key",
                **_VALID_KWARGS,
            )
        )  # no raise
