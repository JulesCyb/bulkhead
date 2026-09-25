"""The MCP server's development-only context provider: identity_id naming, env-based."""

from __future__ import annotations

import uuid

import pytest

from app.config import Settings
from app.mcp import server as mcp_server


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
