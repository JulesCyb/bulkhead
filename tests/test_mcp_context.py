"""The MCP server's development-only context provider (identity_id naming, env-based), tenant
suspension's second refusal point (the session layer's routing read) for the `stdio` fallback
(#106, ADR-0010), and its startup
transport guard (issue #48 / ADR-0005)."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

import app.db.session as session_module
from app.config import Settings
from app.db.session import TenantSuspendedError
from app.mcp import server as mcp_server
from app.token_verifier import set_default_adapter_for_tests
from tests.conftest import FakeControlPlaneReads

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


async def test_resolve_context_no_longer_checks_suspension_itself(monkeypatch):
    """`resolve_context()` (#106) carries no suspension check of its own any more -- a
    per-connection context, however implausible to find suspended (real ones are only ever set by
    `MCPTenantAuthMiddleware` after `resolve_bearer_context` already refused a suspended tenant),
    is simply returned unchanged."""
    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    token = mcp_server._connection_context.set(
        SimpleNamespace(tenant_id=tenant_id, identity_id=identity_id)
    )
    try:
        _install_suspended_tenant(tenant_id)
        ctx = await mcp_server.resolve_context()
    finally:
        mcp_server._connection_context.reset(token)
    assert ctx.tenant_id == tenant_id


async def test_stdio_fallback_tool_call_is_refused_by_the_session_layer_when_suspended(
    monkeypatch,
):
    """Suspension's second refusal point (#106, `app/db/session.py`'s module docstring): the
    `stdio` transport's env-based context (`_context_from_env`) carries no tenant record, so
    nothing refuses it at `resolve_context()` -- the first tool call that actually opens a
    `tenant_session()` (here, the real `document_tools.search_documents`, reached through the
    real `resolve_context()`/`_masked` chain, not a stand-in) hits `tenant_session()`'s own routing
    read and raises there instead, and `_masked` lets `TenantSuspendedError` propagate unmasked."""
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

    async def _raise_suspended(ctx):
        assert ctx.tenant_id == tenant_id
        raise TenantSuspendedError(ctx.tenant_id)

    monkeypatch.setattr(session_module, "_resolve_tenant_alias", _raise_suspended)

    with pytest.raises(TenantSuspendedError):
        await mcp_server.search_documents("query")


def _install_suspended_tenant(tenant_id: uuid.UUID) -> None:
    set_default_adapter_for_tests(FakeControlPlaneReads(auth_settings={tenant_id: (None, True)}))


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

    async def _passing_guard() -> None:
        return None

    monkeypatch.setattr(mcp_server, "run_role_rls_guard", _passing_guard)

    calls: list[str] = []
    monkeypatch.setattr(mcp_server.server, "run", lambda transport: calls.append(transport))

    mcp_server.main()  # must not raise -- check_mcp_mode passes for stdio in dev

    assert calls == ["stdio"]
    # The stdio path's context resolution is unchanged: no connection ever set
    # `_connection_context`, so `resolve_context()` still falls back to the process-wide
    # env identity -- never something only the networked transport should use.
    assert mcp_server._connection_context.get() is None


# --- run_role_rls_guard runs before `main()` ever serves a tool call (issue #81) ---


def _dev_stdio_settings() -> Settings:
    return Settings(
        _env_file=None,
        environment="dev",
        auth_mode="dev-headers",
        mcp_transport="stdio",
        **_VALID_KWARGS,
    )


def test_main_runs_the_role_rls_guard_before_serving(monkeypatch):
    """A fake `run_role_rls_guard` that only records being called proves `main()` runs it, and
    the ordering of `calls` proves it runs before `server.run` -- never a real stdio server ever
    starting inside this test."""
    monkeypatch.setattr(mcp_server, "get_settings", _dev_stdio_settings)
    monkeypatch.setattr(mcp_server, "run_startup_checks", lambda s: None)

    calls: list[str] = []

    async def recording_guard() -> None:
        calls.append("guard")

    monkeypatch.setattr(mcp_server, "run_role_rls_guard", recording_guard)
    monkeypatch.setattr(mcp_server.server, "run", lambda transport: calls.append("serve"))

    mcp_server.main()

    assert calls == ["guard", "serve"]


def test_main_refuses_to_start_when_the_role_rls_guard_fails(monkeypatch):
    """A substituted, failing role/RLS guard makes `main()` refuse to start -- `server.run` (the
    stdio serving loop) must never be reached, and the exception must propagate rather than being
    swallowed."""
    monkeypatch.setattr(mcp_server, "get_settings", _dev_stdio_settings)
    monkeypatch.setattr(mcp_server, "run_startup_checks", lambda s: None)

    async def failing_guard() -> None:
        raise RuntimeError("substituted role/RLS guard failure")

    monkeypatch.setattr(mcp_server, "run_role_rls_guard", failing_guard)

    calls: list[str] = []
    monkeypatch.setattr(mcp_server.server, "run", lambda transport: calls.append(transport))

    with pytest.raises(RuntimeError, match="substituted role/RLS guard failure"):
        mcp_server.main()

    assert calls == []


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
