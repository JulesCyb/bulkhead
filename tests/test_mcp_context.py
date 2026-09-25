"""The MCP server's development-only context provider: identity_id naming, env-based, and its
suspension check (Spec 9 / #69, ADR-0010)."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

import app.tenant_suspension as tenant_suspension_module
from app.config import Settings
from app.mcp import server as mcp_server
from app.tenant_suspension import TenantSuspendedError


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
    `resolve_context()`, which every tool calls instead of `context_provider()` directly, before
    any tool body -- here `search_documents` -- ever runs."""
    tenant_id, identity_id = uuid.uuid4(), uuid.uuid4()
    monkeypatch.setattr(
        mcp_server,
        "context_provider",
        lambda: SimpleNamespace(tenant_id=tenant_id, identity_id=identity_id),
    )

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

    with pytest.raises(TenantSuspendedError):
        await mcp_server.resolve_context()

    async def _boom(*args, **kwargs):
        pytest.fail("search_documents' tool body must not run for a suspended tenant")

    monkeypatch.setattr(mcp_server.document_tools, "search_documents", _boom)
    with pytest.raises(TenantSuspendedError):
        await mcp_server.search_documents("query")
