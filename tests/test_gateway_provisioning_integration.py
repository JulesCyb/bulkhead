"""Real database, real filesystem, faked gateway (Spec 7 / #53, ADR-0009, ADR-0011).

Runs `app.gateway_provisioning.provision_gateway_credential`/`revoke_gateway_credential`, and
`scripts/seed.py`, against an embedded Postgres instance migrated to head (`pgserver`, via
`uv sync --group dbtest`) -- proving the control-plane write (owner role only, #12), the
resolver read-back (#52), and the secret-file lifecycle all work end to end. The gateway's
administrative HTTP interface is never reached over a real network: `GatewayAdminClient` is
always built on `httpx.MockTransport` here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

pgserver = pytest.importorskip("pgserver")

from app.config import ROLE_STATEMENT_TIMEOUT_MS  # noqa: E402
from app.gateway_provisioning import (  # noqa: E402
    GatewayAdminClient,
    GatewayCredentialLimits,
    provision_gateway_credential,
    revoke_gateway_credential,
)

LIMITS = GatewayCredentialLimits(
    spend_ceiling_usd=25.0,
    budget_reset_period="30d",
    requests_per_minute=60,
    tokens_per_minute=100_000,
)


@pytest.fixture(scope="module")
def database_urls():
    """Mirrors tests/test_rls_integration.py's own fixture: app_owner/app roles, migrated to
    head with the real Alembic chain (including #52's 0008_gateway_credential_alias)."""

    from pgserver.postgres_server import POSTGRES_BIN_PATH

    def _psql(server, command: str) -> None:
        subprocess.run(
            [str(POSTGRES_BIN_PATH / "psql"), server.get_uri()],
            input=command.encode(),
            check=True,
            capture_output=True,
        )

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
    from app import config, migration_settings
    from app.db import session as db_session

    monkeypatch.setenv("DATABASE_URL", database_urls["app"])
    monkeypatch.setenv("DATABASE_URL_MIGRATIONS", database_urls["migrations"])
    config.get_settings.cache_clear()
    migration_settings.get_migration_settings.cache_clear()
    db_session._engine = None
    db_session._session_factory = None
    yield
    config.get_settings.cache_clear()
    migration_settings.get_migration_settings.cache_clear()
    db_session._engine = None
    db_session._session_factory = None


async def _seed_tenant(url: str) -> uuid.UUID:
    """A bare tenant row -- as the owner, without a tenant context RLS blocks all rows."""
    engine = create_async_engine(url)
    tenant_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant_id, "name": "Acme"},
        )
    await engine.dispose()
    return tenant_id


def _fake_admin_client(*, key: str = "sk-minted") -> GatewayAdminClient:
    revoke_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/key/generate":
            return httpx.Response(200, json={"key": key})
        if request.url.path == "/key/delete":
            revoke_calls.append(json.loads(request.content)["keys"][0])
            return httpx.Response(200, json={"deleted_keys": revoke_calls})
        return httpx.Response(404)

    http_client = httpx.AsyncClient(
        base_url="http://litellm.internal:4000",
        transport=httpx.MockTransport(handler),
    )
    client = GatewayAdminClient(
        base_url="http://litellm.internal:4000",
        master_key="sk-master-test",
        http_client=http_client,
    )
    client.revoke_calls = revoke_calls  # type: ignore[attr-defined]
    return client


async def test_provision_writes_file_and_alias_is_readable_through_the_resolver(
    app_settings, database_urls, tmp_path
):
    from app.config import Settings
    from app.context import RequestContext
    from app.db.session import tenant_session
    from app.gateway_credentials import resolve_gateway_credential
    from app.repositories.control import ControlRepository

    tenant_id = await _seed_tenant(database_urls["superuser"])
    owner_engine = create_async_engine(database_urls["migrations"])
    settings = Settings(gateway_credentials_dir=str(tmp_path))

    alias = await provision_gateway_credential(
        tenant_id,
        residency="eu",
        limits=LIMITS,
        settings=settings,
        admin_client=_fake_admin_client(key="sk-acme"),
        owner_engine=owner_engine,
    )
    await owner_engine.dispose()

    assert (tmp_path / alias).read_text() == "sk-acme"

    ctx = RequestContext(tenant_id=tenant_id, identity_id=uuid.uuid4())
    async with tenant_session(ctx) as session:
        recorded_alias = await ControlRepository().get_gateway_credential_alias(session, ctx)
        assert recorded_alias == alias

        credential = await resolve_gateway_credential(session, ctx, settings=settings)
        assert credential.get_secret_value() == "sk-acme"


async def test_provisioning_two_tenants_yields_distinct_alias_and_credential(
    app_settings, database_urls, tmp_path
):
    from app.config import Settings

    tenant_a = await _seed_tenant(database_urls["superuser"])
    tenant_b = await _seed_tenant(database_urls["superuser"])
    settings = Settings(gateway_credentials_dir=str(tmp_path))

    owner_engine_a = create_async_engine(database_urls["migrations"])
    alias_a = await provision_gateway_credential(
        tenant_a,
        residency="eu",
        limits=LIMITS,
        settings=settings,
        admin_client=_fake_admin_client(key="sk-a"),
        owner_engine=owner_engine_a,
    )
    await owner_engine_a.dispose()

    owner_engine_b = create_async_engine(database_urls["migrations"])
    alias_b = await provision_gateway_credential(
        tenant_b,
        residency="eu",
        limits=LIMITS,
        settings=settings,
        admin_client=_fake_admin_client(key="sk-b"),
        owner_engine=owner_engine_b,
    )
    await owner_engine_b.dispose()

    assert alias_a != alias_b
    assert (tmp_path / alias_a).read_text() != (tmp_path / alias_b).read_text()


async def test_revoke_calls_gateway_once_removes_file_and_second_revoke_is_noop(
    app_settings, database_urls, tmp_path
):
    from app.config import Settings

    tenant_id = await _seed_tenant(database_urls["superuser"])
    settings = Settings(gateway_credentials_dir=str(tmp_path))

    provision_engine = create_async_engine(database_urls["migrations"])
    alias = await provision_gateway_credential(
        tenant_id,
        residency="eu",
        limits=LIMITS,
        settings=settings,
        admin_client=_fake_admin_client(key="sk-acme"),
        owner_engine=provision_engine,
    )
    await provision_engine.dispose()
    assert (tmp_path / alias).exists()

    revoke_client = _fake_admin_client()
    revoke_engine = create_async_engine(database_urls["migrations"])
    first = await revoke_gateway_credential(
        tenant_id, settings=settings, admin_client=revoke_client, owner_engine=revoke_engine
    )
    await revoke_engine.dispose()

    assert first is True
    assert revoke_client.revoke_calls == ["sk-acme"]  # type: ignore[attr-defined]
    assert not (tmp_path / alias).exists()

    second_client = _fake_admin_client()
    second_engine = create_async_engine(database_urls["migrations"])
    second = await revoke_gateway_credential(
        tenant_id, settings=settings, admin_client=second_client, owner_engine=second_engine
    )
    await second_engine.dispose()

    assert second is False
    assert second_client.revoke_calls == []  # type: ignore[attr-defined]  # never reached the gateway


async def test_seed_script_produces_a_tenant_with_a_working_gateway_credential_on_disk(
    app_settings, database_urls, tmp_path, monkeypatch
):
    """Acceptance criterion: running the seed script against a fresh database produces a tenant
    with a working gateway credential on disk, with no manual step beyond running the script."""
    import scripts.seed as seed
    from app.config import Settings, get_settings

    monkeypatch.setenv("GATEWAY_CREDENTIALS_DIR", str(tmp_path))
    get_settings.cache_clear()
    settings = get_settings()
    assert isinstance(settings, Settings)

    admin_client = _fake_admin_client(key="sk-seeded")
    await seed.main("Acme", "admin@acme.test", settings=settings, admin_client=admin_client)

    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].read_text() == "sk-seeded"

    get_settings.cache_clear()
