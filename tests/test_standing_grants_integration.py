"""Real isolation test for standing grants (ADR-0005, ADR-0007, Spec 5 / #38) against PostgreSQL
+ pgvector (`pgserver`, via `uv sync --group dbtest`). Pattern:
`tests/test_agent_credentials_integration.py`.

Proves what a unit test on `StandingGrantRepository` alone cannot: that the target-role check and
the partial unique index really run against live Postgres state, that the RLS policy from
migration 0031 isolates one tenant's grants from another's, and that the admin-only role gate on
`app/tools/standing_grants.py` really refuses a non-admin caller end to end.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import uuid
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import ROLE_STATEMENT_TIMEOUT_MS  # noqa: E402
from app.context import RequestContext  # noqa: E402
from app.db.session import tenant_session  # noqa: E402
from app.repositories.standing_grants import (  # noqa: E402
    NotAnAgentMembership,
    StandingGrantRepository,
)
from app.tools.standing_grants import (  # noqa: E402
    create_standing_grant,
    list_standing_grants,
    revoke_standing_grant,
)

pgserver = pytest.importorskip("pgserver")


def _psql(server, command: str) -> None:
    """`server.psql` without a shell: pgserver's own version breaks on paths with spaces."""
    from pgserver.postgres_server import POSTGRES_BIN_PATH

    subprocess.run(
        [str(POSTGRES_BIN_PATH / "psql"), server.get_uri()],
        input=command.encode(),
        check=True,
        capture_output=True,
    )


@pytest.fixture(scope="module")
def database_urls():
    """Mirrors `tests/test_rls_integration.py`'s own fixture: app_owner/app roles, migrated to
    head with the real Alembic chain (including #38's 0031_standing_grants)."""
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


async def _seed_tenant(url: str) -> uuid.UUID:
    engine = create_async_engine(url)
    tenant_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, 'Acme')"), {"id": tenant_id}
        )
    await engine.dispose()
    return tenant_id


async def _seed_membership(
    url: str, *, tenant_id: uuid.UUID, role: str
) -> tuple[uuid.UUID, uuid.UUID]:
    """A global identity plus its membership of `role` in `tenant_id`. Returns
    (identity_id, membership_id)."""
    engine = create_async_engine(url)
    identity_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO control.identities (id, issuer, subject) VALUES (:id, 'seed', :sub)"),
            {"id": identity_id, "sub": str(identity_id)},
        )
        membership_id = (
            await conn.execute(
                text(
                    "INSERT INTO memberships (tenant_id, identity_id, role) "
                    "VALUES (:tid, :iid, :role) RETURNING id"
                ),
                {"tid": tenant_id, "iid": identity_id, "role": role},
            )
        ).scalar_one()
    await engine.dispose()
    return identity_id, membership_id


async def test_creating_a_grant_for_a_non_agent_membership_is_refused(app_settings, database_urls):
    """AC1: a membership that does not carry the agent role cannot be granted a standing
    permission."""
    tenant_id = await _seed_tenant(database_urls["superuser"])
    _, admin_membership = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="admin"
    )
    _, member_membership = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="member"
    )

    ctx = RequestContext(tenant_id=tenant_id, identity_id=uuid.uuid4(), roles=frozenset({"admin"}))
    repo = StandingGrantRepository()
    with pytest.raises(NotAnAgentMembership):
        async with tenant_session(ctx) as session:
            await repo.create(
                session,
                ctx,
                agent_membership_id=member_membership,
                tool_name="send_invoice",
                granted_by=admin_membership,
            )


async def test_creating_a_grant_on_behalf_of_a_non_admin_is_refused(app_settings, database_urls):
    """AC2: the same role check Spec 3 (#26) provides refuses a non-admin caller, before any
    data access."""
    tenant_id = await _seed_tenant(database_urls["superuser"])
    _, admin_membership = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="admin"
    )
    _, agent_membership = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="agent"
    )

    for role in ("member", "support", "agent"):
        ctx = RequestContext(tenant_id=tenant_id, identity_id=uuid.uuid4(), roles=frozenset({role}))
        with pytest.raises(PermissionError):
            await create_standing_grant(
                ctx,
                agent_membership_id=agent_membership,
                tool_name="send_invoice",
                granted_by=admin_membership,
            )


async def test_second_active_grant_for_same_tool_is_rejected_a_different_tool_succeeds(
    app_settings, database_urls
):
    """AC3: a partial uniqueness rule rejects a second active grant for the same tenant, agent
    identity, and tool, while a second grant naming a different tool for the same agent identity
    succeeds."""
    tenant_id = await _seed_tenant(database_urls["superuser"])
    _, admin_membership = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="admin"
    )
    _, agent_membership = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="agent"
    )
    ctx = RequestContext(tenant_id=tenant_id, identity_id=uuid.uuid4(), roles=frozenset({"admin"}))
    repo = StandingGrantRepository()

    async with tenant_session(ctx) as session:
        await repo.create(
            session,
            ctx,
            agent_membership_id=agent_membership,
            tool_name="send_invoice",
            granted_by=admin_membership,
        )

    with pytest.raises(IntegrityError):
        async with tenant_session(ctx) as session:
            await repo.create(
                session,
                ctx,
                agent_membership_id=agent_membership,
                tool_name="send_invoice",
                granted_by=admin_membership,
            )

    async with tenant_session(ctx) as session:
        second = await repo.create(
            session,
            ctx,
            agent_membership_id=agent_membership,
            tool_name="delete_document",
            granted_by=admin_membership,
        )
        assert second.tool_name == "delete_document"


async def test_revoking_records_who_and_when_never_deletes_and_lookup_returns_none(
    app_settings, database_urls
):
    """AC4: revoking a grant records the revoking membership and a timestamp on the same row
    (never deletes it), and the very next active-grant lookup for that agent identity and tool
    returns none."""
    tenant_id = await _seed_tenant(database_urls["superuser"])
    _, admin_membership = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="admin"
    )
    _, revoking_admin_membership = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="admin"
    )
    _, agent_membership = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="agent"
    )
    ctx = RequestContext(tenant_id=tenant_id, identity_id=uuid.uuid4(), roles=frozenset({"admin"}))
    repo = StandingGrantRepository()

    async with tenant_session(ctx) as session:
        grant = await repo.create(
            session,
            ctx,
            agent_membership_id=agent_membership,
            tool_name="send_invoice",
            granted_by=admin_membership,
        )
        grant_id = grant.id

    async with tenant_session(ctx) as session:
        active_before = await repo.get_active(
            session, ctx, agent_membership_id=agent_membership, tool_name="send_invoice"
        )
        assert active_before is not None

        revoked = await repo.revoke(
            session, ctx, grant_id=grant_id, revoked_by=revoking_admin_membership
        )
        assert revoked is True

        active_after = await repo.get_active(
            session, ctx, agent_membership_id=agent_membership, tool_name="send_invoice"
        )
        assert active_after is None

    # Never deleted -- still present, directly at the database layer, with who/when recorded.
    engine = create_async_engine(database_urls["superuser"])
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text("SELECT revoked_at, revoked_by FROM standing_grants WHERE id = :id"),
                {"id": grant_id},
            )
        ).one()
    await engine.dispose()
    assert row.revoked_at is not None
    assert row.revoked_by == revoking_admin_membership

    # A second revoke of the same, already-revoked grant is a no-op.
    async with tenant_session(ctx) as session:
        already_revoked = await repo.revoke(
            session, ctx, grant_id=grant_id, revoked_by=revoking_admin_membership
        )
        assert already_revoked is False


async def test_grant_created_for_one_tenant_is_invisible_and_unusable_by_another(
    app_settings, database_urls
):
    """AC5: a standing grant created for one tenant is invisible to, and unusable by, a second
    tenant's session under the standard RLS policy."""
    tenant_a = await _seed_tenant(database_urls["superuser"])
    tenant_b = await _seed_tenant(database_urls["superuser"])
    superuser_url = database_urls["superuser"]
    _, admin_a = await _seed_membership(superuser_url, tenant_id=tenant_a, role="admin")
    _, agent_a = await _seed_membership(superuser_url, tenant_id=tenant_a, role="agent")
    _, admin_b = await _seed_membership(superuser_url, tenant_id=tenant_b, role="admin")

    ctx_a = RequestContext(tenant_id=tenant_a, identity_id=uuid.uuid4(), roles=frozenset({"admin"}))
    repo = StandingGrantRepository()
    async with tenant_session(ctx_a) as session:
        grant = await repo.create(
            session,
            ctx_a,
            agent_membership_id=agent_a,
            tool_name="send_invoice",
            granted_by=admin_a,
        )
        grant_id = grant.id

    ctx_b = RequestContext(tenant_id=tenant_b, identity_id=uuid.uuid4(), roles=frozenset({"admin"}))
    async with tenant_session(ctx_b) as session:
        listing_b = await repo.list_for_tenant(session, ctx_b)
        assert listing_b == []

        active_from_b = await repo.get_active(
            session, ctx_b, agent_membership_id=agent_a, tool_name="send_invoice"
        )
        assert active_from_b is None

        revoked_from_b = await repo.revoke(session, ctx_b, grant_id=grant_id, revoked_by=admin_b)
        assert revoked_from_b is False

        # Directly against the table, bypassing the repository's own WHERE clause entirely.
        row = (
            await session.execute(
                text("SELECT id FROM standing_grants WHERE id = :id"), {"id": grant_id}
            )
        ).first()
        assert row is None

    # Still there and active under its own tenant.
    async with tenant_session(ctx_a) as session:
        still_active = await repo.get_active(
            session, ctx_a, agent_membership_id=agent_a, tool_name="send_invoice"
        )
        assert still_active is not None
        assert still_active.id == grant_id


async def test_listing_shows_granting_membership_and_created_at_for_every_grant(
    app_settings, database_urls
):
    """AC6: listing a tenant's standing grants shows the granting membership and creation time
    for every grant, active and revoked alike."""
    tenant_id = await _seed_tenant(database_urls["superuser"])
    _, admin_membership = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="admin"
    )
    _, agent_membership = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="agent"
    )
    ctx = RequestContext(tenant_id=tenant_id, identity_id=uuid.uuid4(), roles=frozenset({"admin"}))
    repo = StandingGrantRepository()

    async with tenant_session(ctx) as session:
        active_grant = await repo.create(
            session,
            ctx,
            agent_membership_id=agent_membership,
            tool_name="send_invoice",
            granted_by=admin_membership,
        )
        revoked_grant = await repo.create(
            session,
            ctx,
            agent_membership_id=agent_membership,
            tool_name="delete_document",
            granted_by=admin_membership,
        )
        await repo.revoke(session, ctx, grant_id=revoked_grant.id, revoked_by=admin_membership)

        listing = {record.id: record for record in await repo.list_for_tenant(session, ctx)}

    assert listing[active_grant.id].granted_by == admin_membership
    assert listing[active_grant.id].created_at is not None
    assert listing[active_grant.id].revoked_at is None

    assert listing[revoked_grant.id].granted_by == admin_membership
    assert listing[revoked_grant.id].created_at is not None
    assert listing[revoked_grant.id].revoked_at is not None
    assert listing[revoked_grant.id].revoked_by == admin_membership


async def test_admin_tool_functions_create_revoke_and_list_end_to_end(app_settings, database_urls):
    """The tool-layer wrappers (app/tools/standing_grants.py) work end to end for an admin
    caller: create, list, revoke, list again."""
    tenant_id = await _seed_tenant(database_urls["superuser"])
    _, admin_membership = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="admin"
    )
    _, agent_membership = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="agent"
    )
    ctx = RequestContext(tenant_id=tenant_id, identity_id=uuid.uuid4(), roles=frozenset({"admin"}))

    created = await create_standing_grant(
        ctx,
        agent_membership_id=agent_membership,
        tool_name="send_invoice",
        granted_by=admin_membership,
    )
    assert created.revoked_at is None

    listing = await list_standing_grants(ctx)
    assert [record.id for record in listing] == [created.id]

    revoked = await revoke_standing_grant(ctx, grant_id=created.id, revoked_by=admin_membership)
    assert revoked is True

    listing_after = await list_standing_grants(ctx)
    assert listing_after[0].revoked_at is not None
