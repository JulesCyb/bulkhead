"""End-to-end integration test for the agent-identity connection chain (Spec 6 / #49's KNOWN
INTEGRATION GAP), against real PostgreSQL + pgvector (`pgserver`, via `uv sync --group dbtest`).

Before this ticket, #46 created agent identities with issuer `'agent'` and a random subject, #47
minted tokens with `iss` = the tenant's human-IdP issuer (signed with `agent_token_signing_key`),
and `app.token_verifier.verify_tenant_token` looked the identity up by `(iss, sub)` and verified
with the tenant's own key (`jwt_verification_key`) -- so a real agent token could never verify: the
issuer used to mint it never matched the issuer/key pair used to check it. This test proves the
whole chain now works, end to end, against a real database:

    create an agent identity (#46, `control.create_agent_identity`)
    -> issue it a credential (#45, `AgentCredentialRepository.create`)
    -> exchange the credential for a token (#47, `exchange_agent_credential`)
    -> verify that token through the shared module (#44, `verify_tenant_token`) with an
       issuer-aware key source exactly like `app/deps.py::get_key_source` and the MCP transport
       (#49) both use
    -> resolve back to the same agent identity, in the same tenant, with role `agent`.

Pattern: `tests/test_agent_identities_integration.py` /
`tests/test_agent_credentials_integration.py`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import uuid
from urllib.parse import parse_qs, urlparse

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.agent_credential_exchange import exchange_agent_credential
from app.config import ROLE_STATEMENT_TIMEOUT_MS, Settings
from app.context import RequestContext
from app.db.session import tenant_session
from app.deps import get_algorithm_source, get_key_source
from app.jwt_verifier import mint_token
from app.repositories.agent_credentials import AgentCredentialRepository
from app.repositories.agent_identities import AgentIdentityRepository
from app.token_verifier import (
    AGENT_IDENTITY_ISSUER,
    TenantTokenVerificationError,
    set_default_adapter_for_tests,
    verify_tenant_token,
)

pgserver = pytest.importorskip("pgserver")


@pytest.fixture(autouse=True)
def _real_control_plane_reads():
    """This file's tests call `verify_tenant_token` directly against a real, ephemeral Postgres
    cluster and need its real, repository-backed `ControlPlaneReads` -- never the suite-wide fake
    `tests/conftest.py`'s autouse `not_suspended` fixture installs by default (#100). Restores the
    real adapter for the duration of every test in this file; `not_suspended`'s own teardown
    already restores it process-wide afterward regardless."""
    set_default_adapter_for_tests(None)
    yield


# Deliberately distinct from AGENT_IDENTITY_ISSUER -- exercises the exact scenario the gap
# description names: a tenant that has its own human-IdP issuer configured must still verify an
# agent token correctly, because the agent path never consults that setting at all.
HUMAN_ISSUER = "https://idp.example.com"
HUMAN_KEY = "human-idp-verification-key-at-least-32-bytes"
AGENT_SIGNING_KEY = "agent-token-signing-key-at-least-32-bytes-long"

# A real RSA keypair (review finding, Spec 6): a tenant's own IdP signs with an asymmetric
# algorithm in production (RS256/ES256) -- only its *public* key ever lives in this deployment's
# own configuration. Generated once per test process; used by the algorithm-pinning tests below to
# prove RS256 human tokens and HS256 agent tokens both verify correctly in the same process, and
# that neither can be checked against the other's key/algorithm.
_HUMAN_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
HUMAN_RSA_PRIVATE_KEY_PEM = _HUMAN_PRIVATE_KEY.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
).decode("utf-8")
HUMAN_RSA_PUBLIC_KEY_PEM = (
    _HUMAN_PRIVATE_KEY.public_key()
    .public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    .decode("utf-8")
)


def _psql(server, command: str) -> None:
    from pgserver.postgres_server import POSTGRES_BIN_PATH

    subprocess.run(
        [str(POSTGRES_BIN_PATH / "psql"), server.get_uri()],
        input=command.encode(),
        check=True,
        capture_output=True,
    )


@pytest.fixture(scope="module")
def database_urls():
    pgdata = tempfile.mkdtemp(prefix="pgdata-")
    server = pgserver.get_server(pgdata, cleanup_mode="delete")
    sockdir = parse_qs(urlparse(server.get_uri()).query)["host"][0]
    _psql(
        server,
        "CREATE EXTENSION IF NOT EXISTS vector; "
        "CREATE ROLE app_owner LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE; "
        "ALTER SCHEMA public OWNER TO app_owner; "
        "GRANT CREATE ON DATABASE postgres TO app_owner; "
        "CREATE ROLE app LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE; "
        "GRANT USAGE ON SCHEMA public TO app; "
        f"ALTER ROLE app SET statement_timeout = '{ROLE_STATEMENT_TIMEOUT_MS}ms';",
    )
    urls = {
        "migrations": f"postgresql+asyncpg://app_owner@/postgres?host={sockdir}",
        "app": f"postgresql+asyncpg://app@/postgres?host={sockdir}",
        "superuser": f"postgresql+asyncpg://postgres@/postgres?host={sockdir}",
    }
    env = {**os.environ, "DATABASE_URL_MIGRATIONS": urls["migrations"], "DATABASE_URL": urls["app"]}
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"], check=True, env=env, timeout=120
    )
    yield urls
    server.cleanup()


@pytest.fixture
def app_settings(database_urls, monkeypatch):
    from app import config
    from app.db import session as db_session

    monkeypatch.setenv("DATABASE_URL", database_urls["app"])
    monkeypatch.setenv("DATABASE_URL_MIGRATIONS", database_urls["migrations"])
    config.get_settings.cache_clear()
    db_session._engine = None
    db_session._session_factory = None
    yield
    config.get_settings.cache_clear()
    db_session._engine = None
    db_session._session_factory = None


async def _seed_tenant_and_admin(url: str) -> tuple[uuid.UUID, uuid.UUID]:
    """A tenant with its own (deliberately unrelated) human-IdP issuer configured, plus a person
    identity as the admin creating the agent identity -- proving the tenant's own issuer/key never
    enters the agent path even when it is configured."""
    engine = create_async_engine(url)
    tenant_id, admin_id = uuid.uuid4(), uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, 'Acme')"), {"id": tenant_id}
        )
        await conn.execute(
            text("INSERT INTO control.tenants (tenant_id, identity_issuer) VALUES (:id, :issuer)"),
            {"id": tenant_id, "issuer": HUMAN_ISSUER},
        )
        await conn.execute(
            text(
                "INSERT INTO control.identities (id, issuer, subject, kind) "
                "VALUES (:id, :issuer, :sub, 'person')"
            ),
            {"id": admin_id, "issuer": HUMAN_ISSUER, "sub": str(admin_id)},
        )
        await conn.execute(
            text(
                "INSERT INTO memberships (tenant_id, identity_id, role) "
                "VALUES (:tid, :iid, 'admin')"
            ),
            {"tid": tenant_id, "iid": admin_id},
        )
    await engine.dispose()
    return tenant_id, admin_id


def _key_source(issuer: str, kid: str | None) -> str:
    """The same issuer-aware routing `app/deps.py::get_key_source` and the MCP transport (#49)
    both implement, minimal here since only one caller needs it."""
    if issuer == AGENT_IDENTITY_ISSUER:
        return AGENT_SIGNING_KEY
    return HUMAN_KEY


async def test_agent_identity_credential_exchange_and_verification_round_trips(
    app_settings, database_urls
):
    """The gap, closed: create an agent identity, issue it a credential, exchange the credential
    for a token, and verify that token through the shared module -- resolving back to the same
    agent identity, in the same tenant, with role `agent`. Also proves the tenant's own,
    unrelated human-IdP issuer/key (HUMAN_ISSUER/HUMAN_KEY) never enters the agent path at all."""
    tenant_id, admin_id = await _seed_tenant_and_admin(database_urls["superuser"])
    admin_ctx = RequestContext(tenant_id=tenant_id, identity_id=admin_id)

    async with tenant_session(admin_ctx) as session:
        created = await AgentIdentityRepository().create(session, admin_ctx, name="nightly sync")
        issued = await AgentCredentialRepository().create(
            session, admin_ctx, identity_id=created.identity_id, name="nightly sync cred"
        )

    settings = Settings(
        _env_file=None,
        environment="test",
        auth_mode="jwt",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        default_identity_issuer=None,
        jwt_verification_key=HUMAN_KEY,
        jwt_algorithm="HS256",
        agent_token_signing_key=AGENT_SIGNING_KEY,
        agent_token_ttl_seconds=300,
    )

    result = await exchange_agent_credential(
        tenant_id=tenant_id, public_id=issued.public_id, secret=issued.secret, settings=settings
    )

    resolved = await verify_tenant_token(
        result.access_token,
        tenant_id=tenant_id,
        key_source=_key_source,
        default_issuer=None,
        algorithm_source=lambda issuer: ("HS256",),
    )

    assert resolved.identity_id == created.identity_id
    assert resolved.role == "agent"
    assert resolved.issuer == AGENT_IDENTITY_ISSUER
    assert resolved.credential_public_id == issued.public_id


async def test_a_revoked_credentials_token_exchange_fails_closed(app_settings, database_urls):
    tenant_id, admin_id = await _seed_tenant_and_admin(database_urls["superuser"])
    admin_ctx = RequestContext(tenant_id=tenant_id, identity_id=admin_id)

    async with tenant_session(admin_ctx) as session:
        created = await AgentIdentityRepository().create(session, admin_ctx, name="nightly sync")
        issued = await AgentCredentialRepository().create(
            session, admin_ctx, identity_id=created.identity_id, name="nightly sync cred"
        )
        await AgentCredentialRepository().revoke(session, admin_ctx, credential_id=issued.id)

    settings = Settings(
        _env_file=None,
        environment="test",
        auth_mode="jwt",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        default_identity_issuer=None,
        jwt_verification_key=HUMAN_KEY,
        jwt_algorithm="HS256",
        agent_token_signing_key=AGENT_SIGNING_KEY,
        agent_token_ttl_seconds=300,
    )

    from app.agent_credential_exchange import AgentCredentialExchangeError

    with pytest.raises(AgentCredentialExchangeError):
        await exchange_agent_credential(
            tenant_id=tenant_id,
            public_id=issued.public_id,
            secret=issued.secret,
            settings=settings,
        )


def _realistic_settings() -> Settings:
    """A production-shaped configuration (review finding, Spec 6): a real IdP signing human
    tokens with RS256 (only the public key ever configured here) and this application's own agent
    tokens signed with the independent, symmetric AGENT_TOKEN_ALGORITHM/AGENT_TOKEN_SIGNING_KEY --
    the exact combination that, before this fix, broke every agent token the moment JWT_ALGORITHM
    was set to a real IdP's asymmetric algorithm."""
    return Settings(
        _env_file=None,
        environment="test",
        auth_mode="jwt",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        default_identity_issuer=None,
        jwt_verification_key=HUMAN_RSA_PUBLIC_KEY_PEM,
        jwt_algorithm="RS256",
        agent_token_signing_key=AGENT_SIGNING_KEY,
        agent_token_ttl_seconds=300,
    )


async def test_rs256_human_token_and_hs256_agent_token_both_verify_in_the_same_process(
    app_settings, database_urls
):
    """The fix, proven end to end with the real per-issuer key/algorithm pinning
    (`app.deps.get_key_source` / `get_algorithm_source`): a human token signed RS256 by a real IdP
    and an agent token signed HS256 by this application's own exchange both verify correctly
    against one shared Settings object -- the exact production configuration (a real, asymmetric
    IdP algorithm) that used to make every agent token unverifiable."""
    tenant_id, admin_id = await _seed_tenant_and_admin(database_urls["superuser"])
    admin_ctx = RequestContext(tenant_id=tenant_id, identity_id=admin_id)

    async with tenant_session(admin_ctx) as session:
        created = await AgentIdentityRepository().create(session, admin_ctx, name="nightly sync")
        issued = await AgentCredentialRepository().create(
            session, admin_ctx, identity_id=created.identity_id, name="nightly sync cred"
        )

    settings = _realistic_settings()
    key_source = get_key_source(settings)
    algorithm_source = get_algorithm_source(settings)

    human_token = mint_token(
        subject=str(admin_id),
        issuer=HUMAN_ISSUER,
        audience=str(tenant_id),
        signing_key=HUMAN_RSA_PRIVATE_KEY_PEM,
        algorithm="RS256",
        ttl_seconds=300,
    )
    resolved_human = await verify_tenant_token(
        human_token,
        tenant_id=tenant_id,
        key_source=key_source,
        default_issuer=None,
        algorithm_source=algorithm_source,
    )
    assert resolved_human.identity_id == admin_id
    assert resolved_human.role == "admin"
    assert resolved_human.issuer == HUMAN_ISSUER

    agent_result = await exchange_agent_credential(
        tenant_id=tenant_id, public_id=issued.public_id, secret=issued.secret, settings=settings
    )
    resolved_agent = await verify_tenant_token(
        agent_result.access_token,
        tenant_id=tenant_id,
        key_source=key_source,
        default_issuer=None,
        algorithm_source=algorithm_source,
    )
    assert resolved_agent.identity_id == created.identity_id
    assert resolved_agent.role == "agent"
    assert resolved_agent.issuer == AGENT_IDENTITY_ISSUER


async def test_agent_token_resigned_with_the_human_algorithm_and_key_is_rejected(
    app_settings, database_urls
):
    """Algorithm-confusion guard: a token claiming the agent issuer but signed RS256 with the
    human IdP's private key must not verify -- `get_key_source`/`get_algorithm_source` pin the
    agent issuer to the agent secret/HS256 unconditionally, so this token's real (RS256) signature
    can never match what it is checked against."""
    tenant_id, _ = await _seed_tenant_and_admin(database_urls["superuser"])
    settings = _realistic_settings()

    forged = mint_token(
        subject="agent-sub-forged",
        issuer=AGENT_IDENTITY_ISSUER,
        audience=str(tenant_id),
        signing_key=HUMAN_RSA_PRIVATE_KEY_PEM,
        algorithm="RS256",
        ttl_seconds=300,
    )
    with pytest.raises(TenantTokenVerificationError):
        await verify_tenant_token(
            forged,
            tenant_id=tenant_id,
            key_source=get_key_source(settings),
            default_issuer=None,
            algorithm_source=get_algorithm_source(settings),
        )


async def test_human_issuer_token_signed_with_the_agent_secret_is_rejected(
    app_settings, database_urls
):
    """Algorithm-confusion guard, the other direction: a token claiming the tenant's own human
    issuer but signed HS256 with the agent secret must not verify -- the human issuer is pinned to
    JWT_VERIFICATION_KEY/RS256 regardless of what algorithm the token itself was actually signed
    with."""
    tenant_id, _ = await _seed_tenant_and_admin(database_urls["superuser"])
    settings = _realistic_settings()

    forged = mint_token(
        subject="admin-sub-forged",
        issuer=HUMAN_ISSUER,
        audience=str(tenant_id),
        signing_key=AGENT_SIGNING_KEY,
        algorithm="HS256",
        ttl_seconds=300,
    )
    with pytest.raises(TenantTokenVerificationError):
        await verify_tenant_token(
            forged,
            tenant_id=tenant_id,
            key_source=get_key_source(settings),
            default_issuer=None,
            algorithm_source=get_algorithm_source(settings),
        )


async def test_alg_none_token_is_rejected(app_settings, database_urls):
    """`alg=none` (an unsigned token) must never verify, for either issuer -- PyJWT is never
    told to accept it (the `algorithms` allow-list `get_algorithm_source` resolves is always an
    explicit, non-"none" list, see app/config.py's supported-algorithm guard)."""
    import jwt as _pyjwt

    tenant_id, _ = await _seed_tenant_and_admin(database_urls["superuser"])
    settings = _realistic_settings()

    unsigned = _pyjwt.encode(
        {"iss": HUMAN_ISSUER, "sub": "admin-sub", "aud": str(tenant_id)},
        key="",
        algorithm="none",
    )
    with pytest.raises(TenantTokenVerificationError):
        await verify_tenant_token(
            unsigned,
            tenant_id=tenant_id,
            key_source=get_key_source(settings),
            default_issuer=None,
            algorithm_source=get_algorithm_source(settings),
        )
