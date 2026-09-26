"""The MCP server's development-only context provider (identity_id naming, env-based), its
suspension check (Spec 9 / #69, ADR-0010), and its startup transport guard (issue #48 /
ADR-0005)."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

import app.tenant_suspension as tenant_suspension_module
from app.config import Settings
from app.mcp import server as mcp_server
from app.tenant_suspension import TenantSuspendedError

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


async def test_resolve_context_rejects_a_suspended_tenant_before_any_tool_runs(monkeypatch):
    """The MCP connection handler's own, independent suspension check (#69, ADR-0010): raised by
    `resolve_context()`, which every tool calls instead of reading `_connection_context` directly,
    before any tool body -- here `search_documents` -- ever runs."""
    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    token = mcp_server._connection_context.set(
        SimpleNamespace(tenant_id=tenant_id, identity_id=identity_id)
    )
    try:
        _install_suspended_tenant(monkeypatch)

        with pytest.raises(TenantSuspendedError):
            await mcp_server.resolve_context()

        async def _boom(*args, **kwargs):
            pytest.fail("search_documents' tool body must not run for a suspended tenant")

        monkeypatch.setattr(mcp_server.document_tools, "search_documents", _boom)
        with pytest.raises(TenantSuspendedError):
            await mcp_server.search_documents("query")
    finally:
        mcp_server._connection_context.reset(token)


def _install_suspended_tenant(monkeypatch):
    @asynccontextmanager
    async def _fake_control_session():
        yield None

    class _FakeTenantAuthSettingsRepository:
        async def get(self, session, *, tenant_id, default_issuer=None):
            return SimpleNamespace(issuer=default_issuer, suspended=True)

    monkeypatch.setattr(tenant_suspension_module, "control_session", _fake_control_session)
    monkeypatch.setattr(
        tenant_suspension_module,
        "TenantAuthSettingsRepository",
        _FakeTenantAuthSettingsRepository,
    )


# --- resolve_context: per-connection contextvar first, stdio-only env fallback, hard error
# under streamable-http (#89) ---


async def test_resolve_context_prefers_the_connection_context_over_the_env_fallback(monkeypatch):
    """Even under `stdio`, a per-connection context set on `_connection_context` (as a real
    connection would) wins over the env-based fallback -- the fallback is a last resort, not a
    default that shadows a live connection's own identity."""
    monkeypatch.setattr(
        mcp_server,
        "get_settings",
        lambda: Settings(_env_file=None, mcp_transport="stdio", **_VALID_KWARGS),
    )
    conn_tenant_id, conn_identity_id = uuid.uuid4(), uuid.uuid4()
    token = mcp_server._connection_context.set(
        SimpleNamespace(tenant_id=conn_tenant_id, identity_id=conn_identity_id)
    )
    try:
        ctx = await mcp_server.resolve_context()
    finally:
        mcp_server._connection_context.reset(token)

    assert ctx.tenant_id == conn_tenant_id
    assert ctx.identity_id == conn_identity_id


async def test_resolve_context_falls_back_to_env_identity_only_under_stdio(monkeypatch):
    """No per-connection context set, transport is `stdio`: falls back to the process-wide
    `MCP_TENANT_ID`/`MCP_IDENTITY_ID` identity -- the only legitimate use of that fallback
    (local development, ADR-0005)."""
    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    monkeypatch.setattr(
        mcp_server,
        "get_settings",
        lambda: Settings(
            _env_file=None,
            mcp_transport="stdio",
            mcp_tenant_id=str(tenant_id),
            mcp_identity_id=str(identity_id),
            **_VALID_KWARGS,
        ),
    )
    assert mcp_server._connection_context.get() is None

    ctx = await mcp_server.resolve_context()

    assert ctx.tenant_id == tenant_id
    assert ctx.identity_id == identity_id


async def test_resolve_context_never_falls_back_under_streamable_http(monkeypatch):
    """No per-connection context set, transport is `streamable-http`: a hard error, never the
    environment fallback (#89) -- the exact leak `MCPTenantAuthMiddleware`'s contextvar exists to
    prevent."""
    monkeypatch.setattr(
        mcp_server,
        "get_settings",
        lambda: Settings(_env_file=None, mcp_transport="streamable-http", **_VALID_KWARGS),
    )
    assert mcp_server._connection_context.get() is None

    with pytest.raises(RuntimeError, match="per-connection"):
        await mcp_server.resolve_context()


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


def test_main_still_serves_stdio_unchanged_in_development(monkeypatch):
    """Issue #49's acceptance criterion 5: the local development entrypoint (`uv run python -m
    app.mcp.server`) still starts and serves its tool over stdio, with no token involved, exactly
    as it did before the networked transport existed -- `main()` never touches
    `build_streamable_http_app`/`MCPTenantAuthMiddleware` for the stdio transport."""
    settings = Settings(
        _env_file=None,
        environment="dev",
        auth_mode="dev-headers",
        mcp_transport="stdio",
        **_VALID_KWARGS,
    )
    monkeypatch.setattr(mcp_server, "get_settings", lambda: settings)
    monkeypatch.setattr(mcp_server, "run_startup_checks", lambda s: None)

    calls: list[str] = []
    monkeypatch.setattr(mcp_server.server, "run", lambda transport: calls.append(transport))

    mcp_server.main()  # must not raise -- check_mcp_mode passes for stdio in dev

    assert calls == ["stdio"]
    # The stdio path's context resolution is unchanged: no connection ever set
    # `_connection_context`, so `resolve_context()` still falls back to the process-wide
    # env identity -- never something only the networked transport should use.
    assert mcp_server._connection_context.get() is None


def test_check_mcp_mode_raises_for_streamable_http_with_mcp_tenant_id_set(monkeypatch):
    """#89: the leak this closes -- an operator "fixing" the old `context_provider` bug by
    setting `MCP_TENANT_ID`/`MCP_IDENTITY_ID` under `streamable-http` must never be allowed to
    start; that config used to make every authenticated connection act as one fixed identity in
    one fixed tenant, a cross-tenant leak."""
    for env_kwargs in (
        {"mcp_tenant_id": str(uuid.uuid4())},
        {"mcp_identity_id": str(uuid.uuid4())},
        {"mcp_tenant_id": str(uuid.uuid4()), "mcp_identity_id": str(uuid.uuid4())},
    ):
        with pytest.raises(RuntimeError, match="MCP_TENANT_ID|MCP_IDENTITY_ID"):
            mcp_server.check_mcp_mode(
                Settings(
                    _env_file=None,
                    environment="prod",
                    auth_mode="jwt",
                    mcp_transport="streamable-http",
                    jwt_verification_key="a-verification-key",
                    **env_kwargs,
                    **_VALID_KWARGS,
                )
            )


def test_check_mcp_mode_does_not_raise_for_streamable_http_with_a_verifier_configured():
    for environment in ("dev", "test", "prod"):
        mcp_server.check_mcp_mode(
            Settings(
                _env_file=None,
                environment=environment,
                auth_mode="jwt",
                mcp_transport="streamable-http",
                jwt_verification_key="a-verification-key",
                mcp_allowed_hosts="mcp.example.com",
                **_VALID_KWARGS,
            )
        )  # no raise


def test_check_mcp_mode_raises_for_streamable_http_without_allowed_hosts():
    """Issue #116: `build_streamable_http_app` would otherwise pass no `transport_security` to
    the MCP SDK's own app, which auto-enables its localhost-only DNS-rebinding allow-list
    (`host="127.0.0.1"`) -- rejecting every real deployment's Host header. `MCP_ALLOWED_HOSTS`
    (comma-separated, same shape as `CORS_ORIGINS`) must be set before `streamable-http` starts,
    even with a token verifier otherwise fully configured."""
    with pytest.raises(RuntimeError, match="MCP_ALLOWED_HOSTS"):
        mcp_server.check_mcp_mode(
            Settings(
                _env_file=None,
                environment="prod",
                auth_mode="jwt",
                mcp_transport="streamable-http",
                jwt_verification_key="a-verification-key",
                mcp_allowed_hosts="",
                **_VALID_KWARGS,
            )
        )
