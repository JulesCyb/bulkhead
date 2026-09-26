"""ASGI-seam tests for the shared "not found in this tenant" exception family
(`app.repositories.errors.NotFoundInTenant`) and its one registered handler
(`app.main.handle_not_found_in_tenant`).

Finding from the 2026-09-25 review: `NotAnAgentMembership`
(`app/repositories/standing_grants.py`) had the identical "refuse the same way for an unknown id,
a cross-tenant id, or the wrong role" reasoning as `UnknownAgentIdentity`
(`app/repositories/agent_credentials.py`, covered end to end by
`tests/test_agent_identities_api.py`), but no handler was ever registered for it, so it fell
through to the generic 500 handler -- itself a leak (500 vs 404 already tells a caller "this id
exists somewhere" a 404 wouldn't).

`app/tools/standing_grants.py` is not exposed over any HTTP route or MCP tool today (grepped
`app/api/` and `app/mcp/server.py`; only `create_agent_identity`/`issue_agent_credential`/
`list_agent_credentials`/`revoke_agent_credential` and the membership listing are), so there is
no real route to drive an ASGI request through the standing-grants tool. This file instead proves
the mapping the same way `tests/test_error_handling.py` proves the generic-500 and 403 cases: by
installing a search tool that raises the exception in question for every run the one-shot run
route (an existing, real route) prepares (`tests/conftest.py`'s `route_run`), and observing what
the real, registered `app.main` exception handler turns it into over a real ASGI call -- the
tool/handler seam, not a fabricated route.
"""

from __future__ import annotations

import uuid

import httpx
from pydantic_ai.models.test import TestModel

from app.main import app
from app.repositories.agent_credentials import UnknownAgentIdentity
from app.repositories.errors import NotFoundInTenant
from app.repositories.standing_grants import NotAnAgentMembership


def _headers() -> dict[str, str]:
    return {"X-Identity-Id": str(uuid.uuid4())}


def _tenant_path(suffix: str) -> str:
    return f"/v1/t/{uuid.uuid4()}{suffix}"


async def _post_run_that_raises(route_run, exc: Exception) -> httpx.Response:
    async def _raise(ctx, query, limit):
        raise exc

    route_run(TestModel(call_tools=["search_documents"]), search=_raise)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            _tenant_path("/agents/assistant/run"), json={"prompt": "Hi"}, headers=_headers()
        )


# --- The finding itself: NotAnAgentMembership must be a 404, not the generic 500 ---


async def test_not_an_agent_membership_is_404_not_500(route_run):
    response = await _post_run_that_raises(
        route_run, NotAnAgentMembership("membership abc123 does not carry the agent role")
    )

    assert response.status_code == 404, response.text
    assert response.json() == {
        "error": "not_found",
        "message": "No such agent membership in this tenant.",
    }
    # The raw detail (which names the real membership id) never reaches the client.
    assert "abc123" not in response.text


# --- The sibling case (UnknownAgentIdentity) keeps its existing body/behaviour ---


async def test_unknown_agent_identity_keeps_its_existing_404_body(route_run):
    response = await _post_run_that_raises(
        route_run, UnknownAgentIdentity("identity xyz789 has no agent membership in this tenant")
    )

    assert response.status_code == 404, response.text
    assert response.json() == {
        "error": "not_found",
        "message": "No such agent identity in this tenant.",
    }
    assert "xyz789" not in response.text


# --- Both are the same exception family, with the same shared handler ---


def test_both_exceptions_are_notfoundintenant_subclasses():
    assert issubclass(NotAnAgentMembership, NotFoundInTenant)
    assert issubclass(UnknownAgentIdentity, NotFoundInTenant)
    # Distinct public messages -- the family shares a handler, not a message.
    assert NotAnAgentMembership.public_message != UnknownAgentIdentity.public_message


async def test_a_plain_notfoundintenant_subclass_with_no_override_gets_the_base_message(
    route_run,
):
    """A future repository that subclasses `NotFoundInTenant` without setting its own
    `public_message` still gets a clean 404 (the base class's own fixed sentence), never a
    crash -- proving the handler needs no per-exception registration to do the right thing."""

    class SomeFutureNotFound(NotFoundInTenant):
        pass

    response = await _post_run_that_raises(route_run, SomeFutureNotFound("internal detail"))

    assert response.status_code == 404, response.text
    assert response.json() == {"error": "not_found", "message": "Not found in this tenant."}
    assert "internal detail" not in response.text


# --- Happy path unchanged: a normal run still succeeds when nothing raises ---


async def test_happy_path_run_is_unaffected(route_run):
    route_run(TestModel(custom_output_text="hello"))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            _tenant_path("/agents/assistant/run"), json={"prompt": "Hi"}, headers=_headers()
        )

    assert response.status_code == 200, response.text
