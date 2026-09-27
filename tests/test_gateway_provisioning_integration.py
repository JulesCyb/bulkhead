"""Real database, real filesystem, faked gateway (Spec 7 / #53, ADR-0009, ADR-0011).

Runs `app.gateway_provisioning.provision_gateway_credential`/`revoke_gateway_credential` against
the shared embedded Postgres cluster (`tests.support`, issue #96 / spec #90, "A6"), migrated to
head -- proving the control-plane write (owner role only, #12), the resolver read-back (#52), and
the secret-file lifecycle all work end to end. The gateway's administrative HTTP interface is
never reached over a real network: `GatewayAdminClient` is always built on the shared fake,
`tests.support.gateway.fake_gateway_admin_client` (issue #98 / "A6-T3").

The retired `scripts/seed.py`'s own equivalent tests (a tenant with a working credential on disk,
one identity and one admin membership, a second membership attached without touching the first)
now live in `tests/test_operator_tool_integration.py`, exercising the operator tool's `create`
command (Spec 9 / #70) that replaced it.
"""

from __future__ import annotations

import pytest

from app.gateway_provisioning import (
    GatewayCredentialLimits,
    provision_gateway_credential,
    revoke_gateway_credential,
)

pgserver = pytest.importorskip("pgserver")

from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from tests.support import cluster, environment, seed_tenant  # noqa: E402
from tests.support.gateway import fake_gateway_admin_client  # noqa: E402

# `cluster`/`environment` are imported only so pytest can discover them as fixtures from this
# module's namespace -- referenced only by parameter name in the tests below, never called
# directly.
_ = (cluster, environment)

LIMITS = GatewayCredentialLimits(
    spend_ceiling_usd=25.0,
    budget_reset_period="30d",
    requests_per_minute=60,
    tokens_per_minute=100_000,
)


async def test_provision_writes_file_and_alias_is_readable_through_the_resolver(
    environment, tmp_path
):
    from app.config import Settings
    from app.db.session import tenant_record_session
    from app.gateway_credentials import resolve_gateway_credential
    from app.repositories.control import ControlRepository

    tenant = await seed_tenant(environment, via_operator=False)
    owner_engine = create_async_engine(environment.owner_url)
    settings = Settings(gateway_credentials_dir=str(tmp_path))

    alias = await provision_gateway_credential(
        tenant.tenant_id,
        residency="eu",
        limits=LIMITS,
        settings=settings,
        admin_client=fake_gateway_admin_client(key="sk-acme"),
        owner_engine=owner_engine,
    )
    await owner_engine.dispose()

    assert (tmp_path / alias).read_text() == "sk-acme"

    async with tenant_record_session(tenant.tenant_id) as session:
        record = await ControlRepository().get_tenant_record(session, tenant_id=tenant.tenant_id)
    assert record.gateway_credential_alias == alias

    credential = resolve_gateway_credential(record, settings=settings)
    assert credential.get_secret_value() == "sk-acme"


async def test_provisioning_two_tenants_yields_distinct_alias_and_credential(environment, tmp_path):
    from app.config import Settings

    tenant_a = await seed_tenant(environment, via_operator=False)
    tenant_b = await seed_tenant(environment, via_operator=False)
    settings = Settings(gateway_credentials_dir=str(tmp_path))

    owner_engine_a = create_async_engine(environment.owner_url)
    alias_a = await provision_gateway_credential(
        tenant_a.tenant_id,
        residency="eu",
        limits=LIMITS,
        settings=settings,
        admin_client=fake_gateway_admin_client(key="sk-a"),
        owner_engine=owner_engine_a,
    )
    await owner_engine_a.dispose()

    owner_engine_b = create_async_engine(environment.owner_url)
    alias_b = await provision_gateway_credential(
        tenant_b.tenant_id,
        residency="eu",
        limits=LIMITS,
        settings=settings,
        admin_client=fake_gateway_admin_client(key="sk-b"),
        owner_engine=owner_engine_b,
    )
    await owner_engine_b.dispose()

    assert alias_a != alias_b
    assert (tmp_path / alias_a).read_text() != (tmp_path / alias_b).read_text()


async def test_revoke_calls_gateway_once_removes_file_and_second_revoke_is_noop(
    environment, tmp_path
):
    from app.config import Settings

    tenant = await seed_tenant(environment, via_operator=False)
    settings = Settings(gateway_credentials_dir=str(tmp_path))

    provision_engine = create_async_engine(environment.owner_url)
    alias = await provision_gateway_credential(
        tenant.tenant_id,
        residency="eu",
        limits=LIMITS,
        settings=settings,
        admin_client=fake_gateway_admin_client(key="sk-acme"),
        owner_engine=provision_engine,
    )
    await provision_engine.dispose()
    assert (tmp_path / alias).exists()

    revoke_client = fake_gateway_admin_client()
    revoke_engine = create_async_engine(environment.owner_url)
    first = await revoke_gateway_credential(
        tenant.tenant_id, settings=settings, admin_client=revoke_client, owner_engine=revoke_engine
    )
    await revoke_engine.dispose()

    assert first is True
    assert revoke_client.fake.deleted_keys == ["sk-acme"]
    assert not (tmp_path / alias).exists()

    second_client = fake_gateway_admin_client()
    second_engine = create_async_engine(environment.owner_url)
    second = await revoke_gateway_credential(
        tenant.tenant_id, settings=settings, admin_client=second_client, owner_engine=second_engine
    )
    await second_engine.dispose()

    assert second is False
    assert second_client.fake.deleted_keys == []  # never reached the gateway
