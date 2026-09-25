"""ASGI tests for S1-T7 / #17: clean, predictable failure surfaces.

Covers all four acceptance criteria:
- a run that fails mid-way returns only a request id, the real failure text goes to the log
- a caller lacking a required role gets a clean rejection, not an unhandled server error
- the chat endpoint's size cap holds against a chunked body with no declared Content-Length
- interactive API docs are reachable only with development-like settings
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator

import httpx
import pytest

from app.api import agents as agents_module
from app.config import Settings
from app.main import app

SENSITIVE_DETAIL = "duplicate key value violates unique constraint documents_pkey_secret"


def _headers() -> dict[str, str]:
    return {"X-Identity-Id": str(uuid.uuid4())}


def _tenant_path(suffix: str) -> str:
    return f"/v1/t/{uuid.uuid4()}{suffix}"


# --- A run that fails mid-way returns only a request id ---


async def test_failed_run_returns_only_a_request_id(monkeypatch, caplog):
    async def _boom(prompt: str, deps) -> str:
        raise RuntimeError(SENSITIVE_DETAIL)

    monkeypatch.setattr(agents_module, "run_assistant", _boom)
    # raise_app_exceptions=False: Starlette's ServerErrorMiddleware builds the response from our
    # handler *and* re-raises the original exception (so an ASGI server can still log it) —
    # httpx.ASGITransport exists precisely to let a test inspect that response instead of
    # propagating the exception here (see httpx's own docstring for this flag).
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    with caplog.at_level(logging.ERROR):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                _tenant_path("/agents/assistant/run"), json={"prompt": "Hi"}, headers=_headers()
            )

    assert response.status_code == 500
    body = response.json()
    assert SENSITIVE_DETAIL not in response.text
    assert body["request_id"]

    # The real detail is written to the server's own log, tagged with that same id.
    assert any(
        SENSITIVE_DETAIL in record.getMessage() or SENSITIVE_DETAIL in (record.exc_text or "")
        for record in caplog.records
    )
    assert any(body["request_id"] in record.getMessage() for record in caplog.records)


# --- A caller lacking a required role gets a clean rejection, never a 500 ---


async def test_missing_role_ends_in_a_clean_rejection_not_a_crash(monkeypatch):
    async def _forbidden(prompt: str, deps) -> str:
        deps.ctx.require_role("admin")
        return "unreachable"

    monkeypatch.setattr(agents_module, "run_assistant", _forbidden)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            _tenant_path("/agents/assistant/run"),
            json={"prompt": "Hi"},
            headers={**_headers(), "X-Roles": "member"},
        )

    assert response.status_code == 403
    body = response.json()
    assert body["error"] == "forbidden"


async def test_403_names_the_required_role_from_the_typed_attribute_not_the_message(monkeypatch):
    """The handler reads `exc.required_role` (`RoleRequired`, `app/context.py`) rather than
    parsing the exception's message with a regex -- proven by using a `RoleRequired` whose message
    text has nothing in common with the old `role '...' required` shape the regex expected."""
    from app.context import RoleRequired

    async def _forbidden(prompt: str, deps) -> str:
        raise RoleRequired("admin", frozenset({"member"}))

    monkeypatch.setattr(agents_module, "run_assistant", _forbidden)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            _tenant_path("/agents/assistant/run"),
            json={"prompt": "Hi"},
            headers={**_headers(), "X-Roles": "member"},
        )

    assert response.status_code == 403
    body = response.json()
    assert body["error"] == "forbidden"
    assert "admin" in body["message"]


# --- The chat endpoint's size cap holds against a chunked body with no declared length ---


def _chat_body_json(text_len: int) -> bytes:
    import json

    body = {
        "id": "conv-1",
        "trigger": "submit-message",
        "messages": [
            {"id": "m1", "role": "user", "parts": [{"type": "text", "text": "x" * text_len}]}
        ],
    }
    return json.dumps(body).encode()


async def test_chunked_body_over_the_cap_is_rejected_before_full_read():
    from app.api.chat import MAX_BODY_BYTES

    # Well over the real cap, chunked into many small pieces so we can show rejection happens
    # long before the generator (and thus the full body) is drained.
    payload = _chat_body_json(text_len=MAX_BODY_BYTES + 100_000)
    chunk_size = 200
    total_chunks = (len(payload) + chunk_size - 1) // chunk_size
    chunks_yielded: list[int] = []

    async def chunked_body() -> AsyncIterator[bytes]:
        for i in range(0, len(payload), chunk_size):
            chunks_yielded.append(i)
            yield payload[i : i + chunk_size]

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        request = client.build_request(
            "POST",
            _tenant_path("/api/chat"),
            content=chunked_body(),
            headers=_headers(),
        )
        # httpx must not have computed a Content-Length for a streamed body.
        assert "content-length" not in {k.lower() for k in request.headers}
        response = await client.send(request)

    assert response.status_code == 413
    # Rejected well before the generator fully drained.
    assert len(chunks_yielded) < total_chunks


async def test_chunked_body_under_the_cap_still_succeeds(monkeypatch, fake_search, fake_history):
    from pydantic_ai.models.test import TestModel

    from app.agents import assistant as assistant_module
    from app.api import chat as chat_module
    from tests.conftest import resolve_to_model

    monkeypatch.setattr(assistant_module.document_tools, "search_documents", fake_search)
    monkeypatch.setattr(
        assistant_module.conversation_tools, "load_conversation_history", fake_history
    )
    monkeypatch.setattr(
        chat_module,
        "resolve_chat_model",
        resolve_to_model(TestModel(call_tools=["search_documents"])),
    )
    payload = _chat_body_json(text_len=100)

    async def chunked_body() -> AsyncIterator[bytes]:
        chunk_size = 32
        for i in range(0, len(payload), chunk_size):
            yield payload[i : i + chunk_size]

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        request = client.build_request(
            "POST",
            _tenant_path("/api/chat"),
            content=chunked_body(),
            headers=_headers(),
        )
        assert "content-length" not in {k.lower() for k in request.headers}
        response = await client.send(request)

    assert response.status_code == 200, response.text


# --- Interactive API docs: reachable only with development-like settings ---


def _settings(environment: str) -> Settings:
    return Settings(
        _env_file=None,
        environment=environment,
        auth_mode="dev-headers" if environment != "prod" else "jwt",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        # A prod-like environment must configure the networked MCP transport (issue #48 / #49) --
        # the stdio transport's process-wide identity fallback is dev/test-only, guarded by
        # check_mcp_mode the same way AUTH_MODE=dev-headers is.
        mcp_transport="streamable-http" if environment == "prod" else "stdio",
        jwt_verification_key="prod-like-settings-test-verification-key"
        if environment == "prod"
        else None,
    )


def _app_for(monkeypatch: pytest.MonkeyPatch, environment: str):
    """`create_app()` reads `get_settings()` at call time, so patching that name in app.main's
    namespace and building a fresh app is enough to get one built from arbitrary settings,
    without touching the process-wide cached Settings the rest of the suite relies on."""
    from app import main as main_module

    monkeypatch.setattr(main_module, "get_settings", lambda: _settings(environment))
    return main_module.create_app()


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
async def test_docs_reachable_in_development(monkeypatch, path):
    transport = httpx.ASGITransport(app=_app_for(monkeypatch, "dev"))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(path)
    assert response.status_code == 200


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
async def test_docs_unreachable_in_production_like_settings(monkeypatch, path):
    transport = httpx.ASGITransport(app=_app_for(monkeypatch, "prod"))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(path)
    assert response.status_code == 404
