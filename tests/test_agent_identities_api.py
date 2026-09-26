"""ASGI-seam tests for S6-T4 / #46: tenant-admin routes that create agent identities and manage
their credentials.

`app/tools/agent_identities.py` and the repositories it calls are faked here (no real Postgres),
the same pattern `tests/test_memberships_api.py` uses -- the real `control.create_agent_identity`
function, the `kind` column, and RLS have their own coverage in
`tests/test_agent_identities_integration.py`.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx
import pytest

import app.tools.agent_identities as agent_identities_tools_module
import app.tools.memberships as memberships_tools_module
from app.context import RequestContext
from app.main import app
from app.repositories.agent_credentials import (
    CredentialRecord,
    IssuedCredential,
    UnknownAgentIdentity,
)
from app.repositories.agent_identities import AgentIdentity
from app.repositories.memberships import MembershipRecord


@asynccontextmanager
async def _fake_session():
    yield None


@dataclass
class _Credential:
    id: uuid.UUID
    identity_id: uuid.UUID
    name: str
    public_id: str
    secret_hash: str
    created_at: datetime
    revoked_at: datetime | None = None
    last_used_at: datetime | None = None


@dataclass
class _World:
    """In-memory stand-in for the tenant-scoped rows a real Postgres session (with RLS) would
    otherwise give us -- keyed by tenant, so cross-tenant isolation is exactly "a different key
    in this dict", matching what the real RLS policy enforces at the database layer."""

    memberships: dict[uuid.UUID, list[MembershipRecord]] = field(default_factory=dict)
    credentials: dict[uuid.UUID, list[_Credential]] = field(default_factory=dict)


def _install_fakes(monkeypatch, world: _World):
    class FakeAgentIdentityRepository:
        async def create(self, session, ctx: RequestContext, *, name: str) -> AgentIdentity:
            identity_id = uuid.uuid4()
            membership_id = uuid.uuid4()
            record = MembershipRecord(
                id=membership_id,
                identity_id=identity_id,
                role="agent",
                created_at=datetime.now(UTC),
            )
            world.memberships.setdefault(ctx.tenant_id, []).append(record)
            return AgentIdentity(identity_id=identity_id, membership_id=membership_id)

    class FakeAgentCredentialRepository:
        async def create(
            self, session, ctx: RequestContext, *, identity_id: uuid.UUID, name: str
        ) -> IssuedCredential:
            # Mirrors the real `AgentCredentialRepository.create`'s own invariant (review of
            # #46): `identity_id` must carry an `agent`-role membership in `ctx.tenant_id`,
            # checked against this fake's own tenant-keyed dict the same way RLS would scope a
            # real query -- an identity from another tenant's dict entry, or a person's
            # membership in this tenant's own entry, is exactly as invisible here as it would be
            # to the real repository.
            role = next(
                (
                    m.role
                    for m in world.memberships.get(ctx.tenant_id, [])
                    if m.identity_id == identity_id
                ),
                None,
            )
            if role != "agent":
                raise UnknownAgentIdentity(
                    f"identity {identity_id} has no agent membership in this tenant"
                )
            cred = _Credential(
                id=uuid.uuid4(),
                identity_id=identity_id,
                name=name,
                public_id=f"agt_{uuid.uuid4().hex[:16]}",
                secret_hash="unused-in-fake",
                created_at=datetime.now(UTC),
            )
            world.credentials.setdefault(ctx.tenant_id, []).append(cred)
            return IssuedCredential(
                id=cred.id,
                identity_id=cred.identity_id,
                name=cred.name,
                public_id=cred.public_id,
                secret="plaintext-secret-shown-exactly-once",
                created_at=cred.created_at,
            )

        async def list_for_tenant(self, session, ctx: RequestContext) -> list[CredentialRecord]:
            return [
                CredentialRecord(
                    id=c.id,
                    identity_id=c.identity_id,
                    name=c.name,
                    public_id=c.public_id,
                    created_at=c.created_at,
                    revoked_at=c.revoked_at,
                    last_used_at=c.last_used_at,
                )
                for c in world.credentials.get(ctx.tenant_id, [])
            ]

        async def revoke(self, session, ctx: RequestContext, *, credential_id: uuid.UUID) -> bool:
            for c in world.credentials.get(ctx.tenant_id, []):
                if c.id == credential_id and c.revoked_at is None:
                    c.revoked_at = datetime.now(UTC)
                    return True
            return False

    class _FakeMembershipListingRepository:
        """Fakes `app.tools.memberships.MembershipRepository.list_for_tenant` -- unrelated to (and
        not migrated by) #100's control-plane-reads adapter, which only covers the narrower
        `get_role` read `app.token_verifier` uses."""

        async def list_for_tenant(self, session, ctx: RequestContext):
            return world.memberships.get(ctx.tenant_id, [])

    monkeypatch.setattr(
        agent_identities_tools_module, "tenant_session", lambda ctx: _fake_session()
    )
    monkeypatch.setattr(
        agent_identities_tools_module, "AgentIdentityRepository", FakeAgentIdentityRepository
    )
    monkeypatch.setattr(
        agent_identities_tools_module, "AgentCredentialRepository", FakeAgentCredentialRepository
    )
    monkeypatch.setattr(memberships_tools_module, "tenant_session", lambda ctx: _fake_session())
    monkeypatch.setattr(
        memberships_tools_module, "MembershipRepository", _FakeMembershipListingRepository
    )


def _headers(identity_id: uuid.UUID, roles: str) -> dict[str, str]:
    return {"X-Identity-Id": str(identity_id), "X-Roles": roles}


async def _client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def _create_identity(tenant_id, headers, name="nightly-sync"):
    async with await _client() as client:
        return await client.post(
            f"/v1/t/{tenant_id}/agent-identities", json={"name": name}, headers=headers
        )


async def _issue_credential(tenant_id, identity_id, headers, name="prod-key"):
    async with await _client() as client:
        return await client.post(
            f"/v1/t/{tenant_id}/agent-identities/{identity_id}/credentials",
            json={"name": name},
            headers=headers,
        )


async def _list_credentials(tenant_id, headers):
    async with await _client() as client:
        return await client.get(f"/v1/t/{tenant_id}/agent-credentials", headers=headers)


async def _revoke_credential(tenant_id, credential_id, headers):
    async with await _client() as client:
        return await client.post(
            f"/v1/t/{tenant_id}/agent-credentials/{credential_id}/revoke", headers=headers
        )


async def _list_memberships(tenant_id, headers):
    async with await _client() as client:
        return await client.get(f"/v1/t/{tenant_id}/memberships", headers=headers)


# --- AC1: creating an agent identity makes it appear in the membership listing with role agent ---


async def test_admin_creates_agent_identity_and_it_appears_in_membership_listing(monkeypatch):
    world = _World()
    _install_fakes(monkeypatch, world)
    tenant_id = uuid.uuid4()
    admin_id = uuid.uuid4()
    headers = _headers(admin_id, "admin")

    create_resp = await _create_identity(tenant_id, headers)
    assert create_resp.status_code == 200, create_resp.text
    identity_id = create_resp.json()["identity_id"]
    assert identity_id

    listing = await _list_memberships(tenant_id, headers)
    assert listing.status_code == 200
    memberships = listing.json()["memberships"]
    matching = [m for m in memberships if m["identity_id"] == identity_id]
    assert len(matching) == 1
    assert matching[0]["role"] == "agent"


# --- AC2: issuing a credential returns the secret once; listing never contains it ---


async def test_admin_issues_credential_secret_shown_once_never_in_listing(monkeypatch):
    world = _World()
    _install_fakes(monkeypatch, world)
    tenant_id = uuid.uuid4()
    admin_id = uuid.uuid4()
    headers = _headers(admin_id, "admin")

    identity_id = (await _create_identity(tenant_id, headers)).json()["identity_id"]

    issue_resp = await _issue_credential(tenant_id, identity_id, headers, name="nightly sync")
    assert issue_resp.status_code == 200, issue_resp.text
    issued = issue_resp.json()
    assert issued["secret"] == "plaintext-secret-shown-exactly-once"
    assert issued["name"] == "nightly sync"
    credential_id = issued["id"]

    listing = await _list_credentials(tenant_id, headers)
    assert listing.status_code == 200
    body = listing.json()["credentials"]
    assert len(body) == 1
    record = body[0]
    assert record["id"] == credential_id
    assert record["name"] == "nightly sync"
    assert "created_at" in record
    assert record["revoked_at"] is None
    assert record["last_used_at"] is None
    assert "secret" not in record
    assert "secret_hash" not in record
    assert "plaintext-secret-shown-exactly-once" not in listing.text


# --- AC3: revoking a credential is reflected immediately in the listing ---


async def test_admin_revokes_credential_reflected_in_listing(monkeypatch):
    world = _World()
    _install_fakes(monkeypatch, world)
    tenant_id = uuid.uuid4()
    admin_id = uuid.uuid4()
    headers = _headers(admin_id, "admin")

    identity_id = (await _create_identity(tenant_id, headers)).json()["identity_id"]
    credential_id = (await _issue_credential(tenant_id, identity_id, headers)).json()["id"]

    before = (await _list_credentials(tenant_id, headers)).json()["credentials"][0]
    assert before["revoked_at"] is None

    revoke_resp = await _revoke_credential(tenant_id, credential_id, headers)
    assert revoke_resp.status_code == 200, revoke_resp.text
    assert revoke_resp.json()["revoked"] is True

    after = (await _list_credentials(tenant_id, headers)).json()["credentials"][0]
    assert after["revoked_at"] is not None


# --- AC4: a non-admin membership gets 403, never 500, on all four actions ---


@pytest.mark.parametrize("role", ["member", "support", "agent"])
async def test_non_admin_gets_403_on_all_four_actions(monkeypatch, role):
    world = _World()
    _install_fakes(monkeypatch, world)
    tenant_id = uuid.uuid4()
    admin_id = uuid.uuid4()
    non_admin_headers = _headers(uuid.uuid4(), role)

    # Seed one identity/credential as admin so revoke/issue have a real target to attempt against.
    identity_id = (await _create_identity(tenant_id, _headers(admin_id, "admin"))).json()[
        "identity_id"
    ]
    credential_id = (
        await _issue_credential(tenant_id, identity_id, _headers(admin_id, "admin"))
    ).json()["id"]

    create_resp = await _create_identity(tenant_id, non_admin_headers)
    issue_resp = await _issue_credential(tenant_id, identity_id, non_admin_headers)
    list_resp = await _list_credentials(tenant_id, non_admin_headers)
    revoke_resp = await _revoke_credential(tenant_id, credential_id, non_admin_headers)

    for response in (create_resp, issue_resp, list_resp, revoke_resp):
        assert response.status_code == 403, response.text
        assert response.json()["error"] == "forbidden"


# --- AC5: an identity/credential created under one tenant never appears in another's listing ---


async def test_identity_and_credential_are_isolated_per_tenant(monkeypatch):
    world = _World()
    _install_fakes(monkeypatch, world)
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    admin_a = _headers(uuid.uuid4(), "admin")
    admin_b = _headers(uuid.uuid4(), "admin")

    identity_id = (await _create_identity(tenant_a, admin_a)).json()["identity_id"]
    await _issue_credential(tenant_a, identity_id, admin_a)

    memberships_b = (await _list_memberships(tenant_b, admin_b)).json()["memberships"]
    assert all(m["identity_id"] != identity_id for m in memberships_b)

    credentials_b = (await _list_credentials(tenant_b, admin_b)).json()["credentials"]
    assert credentials_b == []

    # Tenant A still sees its own data, unaffected by tenant B ever being queried.
    memberships_a = (await _list_memberships(tenant_a, admin_a)).json()["memberships"]
    assert any(m["identity_id"] == identity_id for m in memberships_a)
    credentials_a = (await _list_credentials(tenant_a, admin_a)).json()["credentials"]
    assert len(credentials_a) == 1


# --- Review finding (#46, 2026-09-25): issuing a credential must check identity_id is an agent
# identity belonging to the caller's own tenant, not only that the caller is an admin ---


async def test_issuing_credential_for_another_tenants_agent_identity_is_404_like_unknown_id(
    monkeypatch,
):
    """An admin of tenant A must not be able to mint a credential naming tenant B's agent
    identity -- and the response must be indistinguishable from asking about an id that does not
    exist anywhere at all."""
    world = _World()
    _install_fakes(monkeypatch, world)
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    admin_a = _headers(uuid.uuid4(), "admin")
    admin_b = _headers(uuid.uuid4(), "admin")

    other_tenants_agent_id = (await _create_identity(tenant_b, admin_b)).json()["identity_id"]
    unknown_id = str(uuid.uuid4())

    cross_tenant_resp = await _issue_credential(tenant_a, other_tenants_agent_id, admin_a)
    unknown_resp = await _issue_credential(tenant_a, unknown_id, admin_a)

    for response in (cross_tenant_resp, unknown_resp):
        assert response.status_code == 404, response.text
        assert response.json() == {
            "error": "not_found",
            "message": "No such agent identity in this tenant.",
        }
    # And the same fixed body for both -- nothing distinguishes "exists, wrong tenant" from
    # "doesn't exist anywhere".
    assert cross_tenant_resp.json() == unknown_resp.json()

    # Tenant B's own credential listing is untouched: no row was ever written for its identity.
    credentials_b = (await _list_credentials(tenant_b, admin_b)).json()["credentials"]
    assert credentials_b == []


async def test_issuing_credential_for_a_person_identity_in_own_tenant_is_404(monkeypatch):
    """A membership that exists in the caller's own tenant but is not `agent`-role (a person)
    must be refused the same way an unknown id is -- never distinguished from it."""
    world = _World()
    _install_fakes(monkeypatch, world)
    tenant_id = uuid.uuid4()
    admin_headers = _headers(uuid.uuid4(), "admin")
    person_identity_id = uuid.uuid4()
    world.memberships.setdefault(tenant_id, []).append(
        MembershipRecord(
            id=uuid.uuid4(),
            identity_id=person_identity_id,
            role="member",
            created_at=datetime.now(UTC),
        )
    )

    resp = await _issue_credential(tenant_id, person_identity_id, admin_headers)

    assert resp.status_code == 404, resp.text
    assert resp.json() == {
        "error": "not_found",
        "message": "No such agent identity in this tenant.",
    }
    assert (await _list_credentials(tenant_id, admin_headers)).json()["credentials"] == []


async def test_issuing_credential_for_the_tenants_own_agent_identity_still_works(monkeypatch):
    """Happy path unchanged: an agent identity created and issued a credential in the same
    tenant still succeeds exactly as before."""
    world = _World()
    _install_fakes(monkeypatch, world)
    tenant_id = uuid.uuid4()
    admin_headers = _headers(uuid.uuid4(), "admin")

    identity_id = (await _create_identity(tenant_id, admin_headers)).json()["identity_id"]
    resp = await _issue_credential(tenant_id, identity_id, admin_headers, name="prod-key")

    assert resp.status_code == 200, resp.text
    assert resp.json()["secret"] == "plaintext-secret-shown-exactly-once"
