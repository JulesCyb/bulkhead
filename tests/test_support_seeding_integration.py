"""Tests for the test support package itself (issue #96 / spec #90, "A6"; dedicated-tier
seeding is issue #97 / "A6-T2"; seeding through the operator commands and the shared gateway fake
is issue #98 / "A6-T3").

Proves the guarantees every later integration test in this suite leans on: that `seed_tenant`
really isolates two tenants' documents and memberships through the real session layer (not just
by construction) for both a pooled and a dedicated tenant; that a dedicated tenant's own rows
are physically absent from the pooled database, not merely policy-hidden; that its NOT-NULL-
column guard really fires when a table grows a column the seeder does not know about, instead of
failing with a cryptic constraint violation or silently inserting a wrong default; that the
`environment` fixture's secret directories are genuinely temporary; and (#98) that a seeded
tenant's gateway credential is real (readable through the real credential reader), that
suspending a seeded tenant through the fixture is really enforced by the session layer, and that
the shared gateway fake really recorded what `create_tenant` asked it to mint.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

pgserver = pytest.importorskip("pgserver")

from app.db.session import TenantSuspendedError, get_engine, tenant_session  # noqa: E402
from app.gateway_credentials import read_gateway_credential  # noqa: E402
from app.repositories.documents import DocumentRepository  # noqa: E402
from app.repositories.memberships import MembershipRepository  # noqa: E402
from tests.support import (  # noqa: E402
    UnknownNotNullColumnError,
    assert_known_not_null_columns,
    cluster,
    environment,
    seed_tenant,
    vector_literal,
)
from tests.support.cluster import environment as environment_fixture  # noqa: E402

# `cluster`/`environment` are imported only so pytest can discover them as fixtures from this
# module's namespace (the same "import a fixture" pattern every migrated integration file uses
# below) -- referenced only by parameter name in the tests, never called directly.
_ = (cluster, environment)


async def test_seeding_two_tenants_isolates_documents_and_memberships(environment):
    """AC (spec #90): `seed_tenant()` seeds two tenants; a context of one sees only its own
    documents and memberships through the real session layer."""
    tenant_a = await seed_tenant(environment, name="A", roles=["admin", "member"], documents=1)
    tenant_b = await seed_tenant(environment, name="B", roles=["admin"], documents=1)

    query = [0.0] * 1536
    query[0] = 1.0

    async with tenant_session(tenant_a.ctx("admin")) as session:
        docs_a = await DocumentRepository().search(session, query, limit=10)
        memberships_a = await MembershipRepository().list_for_tenant(session, tenant_a.ctx("admin"))

    async with tenant_session(tenant_b.ctx("admin")) as session:
        docs_b = await DocumentRepository().search(session, query, limit=10)
        memberships_b = await MembershipRepository().list_for_tenant(session, tenant_b.ctx("admin"))

    assert {d.id for d in docs_a} == set(tenant_a.document_ids)
    assert {d.id for d in docs_b} == set(tenant_b.document_ids)
    assert not ({d.id for d in docs_a} & {d.id for d in docs_b})

    assert {m.identity_id for m in memberships_a} == set(tenant_a.identities.values())
    assert {m.identity_id for m in memberships_b} == set(tenant_b.identities.values())
    assert not ({m.identity_id for m in memberships_a} & {m.identity_id for m in memberships_b})


async def test_unknown_not_null_column_guard_fires(cluster):
    """AC (spec #90): seeding fails loudly on an unknown not-null column -- proven here directly
    against `assert_known_not_null_columns` and a throwaway temp table, rather than waiting for
    a real migration to add one to a table `seed_tenant` actually writes to."""
    engine = create_async_engine(cluster.owner_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "CREATE TEMP TABLE support_guard_probe ("
                    "id uuid PRIMARY KEY, surprise text NOT NULL)"
                )
            )
            with pytest.raises(UnknownNotNullColumnError) as exc_info:
                await assert_known_not_null_columns(
                    conn, {"support_guard_probe": frozenset({"id"})}
                )
            assert "surprise" in str(exc_info.value)
    finally:
        await engine.dispose()


def test_vector_literal_is_1536_dimensional():
    literal = vector_literal(0.5)
    assert literal.count(",") == 1535


async def test_dedicated_seeded_tenants_rows_are_unreachable_through_the_pooled_engine(
    environment,
):
    """AC (#97): "A dedicated seeded tenant's rows are unreachable through the pooled engine."
    Not merely policy-hidden: forcing the tenant's own context directly against the pooled
    engine (bypassing routing entirely) still returns zero rows, because the tenant's data was
    never written to the pooled database -- only its control-plane bookkeeping stub was."""
    tenant = await seed_tenant(
        environment, name="Dedicated", roles=["admin"], documents=1, isolation_tier="dedicated"
    )
    assert tenant.database_alias is not None
    assert tenant.database is not None
    assert tenant.owner_url == tenant.database.owner_url

    pooled_engine = get_engine()
    async with pooled_engine.connect() as conn:
        async with conn.begin():
            await conn.execute(
                text("SELECT set_config('app.tenant_id', :tid, true)"),
                {"tid": str(tenant.tenant_id)},
            )
            memberships = (await conn.execute(text("SELECT id FROM memberships"))).scalars().all()
            documents_ = (await conn.execute(text("SELECT id FROM documents"))).scalars().all()

    assert memberships == []
    assert documents_ == []


async def test_dedicated_seeded_tenant_ctx_routes_through_tenant_session_to_its_own_database(
    environment,
):
    """AC (#97): "its `ctx()` routes through `tenant_session` to the dedicated database" --
    proven against the real routing seam, not just by inspecting `SeededTenant.database`."""
    tenant = await seed_tenant(environment, roles=["admin"], isolation_tier="dedicated")

    async with tenant_session(tenant.ctx("admin")) as session:
        assert session.get_bind() is not get_engine().sync_engine
        memberships = await MembershipRepository().list_for_tenant(session, tenant.ctx("admin"))

    assert {m.identity_id for m in memberships} == set(tenant.identities.values())


async def test_environment_fixture_points_secret_dirs_outside_the_repo(environment):
    """AC (#97): "no test writes into the repo's secrets path." Checked against the exact
    resolution `app/db/engine_registry.py`/`scripts/migrate.py` use, not a copy of it."""
    from app.db.engine_registry import _secrets_dir
    from scripts.migrate import _migrations_secrets_dir

    repo_root = Path(__file__).resolve().parent.parent
    secrets_dir = _secrets_dir()
    migrations_dir = _migrations_secrets_dir()

    assert secrets_dir.is_dir()
    assert migrations_dir.is_dir()
    assert repo_root not in (secrets_dir, *secrets_dir.parents)
    assert repo_root not in (migrations_dir, *migrations_dir.parents)


def test_environment_fixture_removes_secret_dirs_on_teardown(cluster, monkeypatch):
    """AC (#97): "Secret directories are temporary and cleaned up." Drives `environment`'s own
    generator directly (`__wrapped__`, the underlying function pytest's fixture decorator
    wraps) so the test can observe both sides of its teardown in one place, rather than only
    trusting that no error was raised."""
    gen = environment_fixture.__wrapped__(cluster, monkeypatch)
    next(gen)

    secrets_dir = os.environ["TENANT_DB_SECRETS_DIR"]
    migrations_dir = os.environ["TENANT_DB_MIGRATIONS_SECRETS_DIR"]
    assert os.path.isdir(secrets_dir)
    assert os.path.isdir(migrations_dir)

    with pytest.raises(StopIteration):
        next(gen)

    assert not os.path.exists(secrets_dir)
    assert not os.path.exists(migrations_dir)


async def test_dedicated_seeding_without_the_environment_fixture_fails_closed(cluster):
    """A `seed_tenant(isolation_tier="dedicated")` call given the bare `cluster` fixture (no
    secrets directories wired) must refuse outright rather than silently falling back to
    `app/db/engine_registry.py`/`scripts/migrate.py`'s production secrets paths."""
    with pytest.raises(RuntimeError, match="TENANT_DB_MIGRATIONS_SECRETS_DIR"):
        await seed_tenant(cluster, isolation_tier="dedicated")


async def test_pooled_seeding_via_operator_without_the_environment_fixture_fails_closed(cluster):
    """The same fail-closed guarantee, for `via_operator=True`'s own secrets directory
    (`GATEWAY_CREDENTIALS_DIR`) -- given the bare `cluster` fixture, seeding must refuse rather
    than silently write a minted credential's secret file to `app.config.Settings`'s production
    default (`/run/secrets`)."""
    with pytest.raises(RuntimeError, match="GATEWAY_CREDENTIALS_DIR"):
        await seed_tenant(cluster)


async def test_seeded_tenants_credential_alias_resolves_to_a_readable_secret_file(environment):
    """AC (#98): "A seeded tenant's credential alias resolves to a readable secret file through
    the real credential reader" -- `app.gateway_credentials.read_gateway_credential`, with no
    settings override, reading whatever `GATEWAY_CREDENTIALS_DIR` the `environment` fixture
    pointed at."""
    tenant = await seed_tenant(environment, roles=["admin"])

    assert tenant.gateway_credential_alias is not None
    credential = read_gateway_credential(tenant.gateway_credential_alias)
    assert credential.get_secret_value() == tenant.gateway_admin_client.fake.key


async def test_suspending_a_seeded_tenant_makes_the_session_router_refuse_it(environment):
    """AC (#98): "Suspending a seeded tenant makes the session router refuse it" -- through
    `SeededTenant.suspend()` (`app.operator.suspend.set_tenant_suspended`), observed against the
    real `tenant_session()`, not by reading back `control.tenants.suspended_at` ourselves."""
    tenant = await seed_tenant(environment, roles=["admin"])

    async with tenant_session(tenant.ctx("admin")):
        pass  # not suspended yet -- a real session opens without error

    result = await tenant.suspend()
    assert result.changed is True

    with pytest.raises(TenantSuspendedError):
        async with tenant_session(tenant.ctx("admin")):
            pass

    unsuspend_result = await tenant.unsuspend()
    assert unsuspend_result.changed is True

    async with tenant_session(tenant.ctx("admin")):
        pass  # restored -- a real session opens without error again


async def test_the_shared_gateway_fake_records_the_mint_requests_models(environment):
    """AC (#98): "The shared fake records the mint request's models" -- the exact `models` list
    `create_tenant` asked the gateway to mint the credential for, taken from the tenant's own
    residency's model allow-list (`config/residency.toml`)."""
    tenant = await seed_tenant(environment, roles=["admin"], residency="eu")

    fake = tenant.gateway_admin_client.fake
    assert fake.generate_calls == 1
    assert fake.generate_requests[-1]["models"] == ["claude-eu", "embeddings"]
    assert fake.generate_requests[-1]["metadata"]["tenant_id"] == str(tenant.tenant_id)
