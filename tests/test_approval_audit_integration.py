"""Real isolation test for the approval audit trail (ADR-0007, Spec 5 / #39) against PostgreSQL
+ pgvector (`pgserver`, via `uv sync --group dbtest`). Pattern:
`tests/test_standing_grants_integration.py`.

Proves what a unit test on `ApprovalAuditRepository` alone cannot: that writing one record for
each of the seven milestone kinds really succeeds against live Postgres, that the RLS policy from
migration 0035 isolates one tenant's audit trail from another's, that the `app` role's own grants
(SELECT, INSERT -- no UPDATE/DELETE) make append-only a database-enforced fact and not just a
convention, and that a tenant-scoped read of one pending action's or standing grant's milestones
comes back most-recent-first.
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
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import ROLE_STATEMENT_TIMEOUT_MS  # noqa: E402
from app.context import RequestContext  # noqa: E402
from app.db.session import tenant_session  # noqa: E402
from app.repositories.approval_audit import (  # noqa: E402
    APPROVED,
    DENIED_FOR_LACK_OF_GRANT,
    EXECUTED,
    EXPIRED,
    FAILED_TO_EXECUTE,
    KINDS,
    REFUSED,
    REQUESTED,
    ApprovalAuditRepository,
    InvalidAuditEventKind,
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
    """Mirrors `tests/test_standing_grants_integration.py`'s own fixture: app_owner/app roles,
    migrated to head with the real Alembic chain (including #39's 0035_approval_audit_events)."""
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


async def test_all_seven_milestone_kinds_can_be_recorded(app_settings, database_urls):
    """AC1: writing one audit record for each of the seven milestone kinds succeeds, and each row
    names the acting membership and, where applicable, the specific pending action or standing
    grant it came from."""
    assert KINDS == {
        REQUESTED,
        APPROVED,
        REFUSED,
        EXPIRED,
        DENIED_FOR_LACK_OF_GRANT,
        EXECUTED,
        FAILED_TO_EXECUTE,
    }

    tenant_id = await _seed_tenant(database_urls["superuser"])
    _, member_membership = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="member"
    )
    _, agent_membership = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="agent"
    )
    ctx = RequestContext(tenant_id=tenant_id, identity_id=uuid.uuid4(), roles=frozenset({"member"}))
    repo = ApprovalAuditRepository()
    pending_action_id = uuid.uuid4()
    standing_grant_id = uuid.uuid4()

    async with tenant_session(ctx) as session:
        for kind, actor, pa_id, sg_id in (
            (REQUESTED, member_membership, pending_action_id, None),
            (APPROVED, member_membership, pending_action_id, None),
            (REFUSED, member_membership, pending_action_id, None),
            (EXPIRED, member_membership, pending_action_id, None),
            (DENIED_FOR_LACK_OF_GRANT, agent_membership, None, None),
            (EXECUTED, agent_membership, None, standing_grant_id),
            (FAILED_TO_EXECUTE, agent_membership, None, standing_grant_id),
        ):
            record = await repo.record(
                session,
                ctx,
                kind=kind,
                tool_name="send_invoice",
                actor_membership_id=actor,
                pending_action_id=pa_id,
                standing_grant_id=sg_id,
            )
            assert record.kind == kind
            assert record.actor_membership_id == actor
            assert record.pending_action_id == pa_id
            assert record.standing_grant_id == sg_id


async def test_recording_an_unknown_kind_is_refused(app_settings, database_urls):
    """`record()` refuses any kind outside the seven milestones before writing a row."""
    tenant_id = await _seed_tenant(database_urls["superuser"])
    _, member_membership = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="member"
    )
    ctx = RequestContext(tenant_id=tenant_id, identity_id=uuid.uuid4(), roles=frozenset({"member"}))
    repo = ApprovalAuditRepository()

    async with tenant_session(ctx) as session:
        with pytest.raises(InvalidAuditEventKind):
            await repo.record(
                session,
                ctx,
                kind="always_allow",
                tool_name="send_invoice",
                actor_membership_id=member_membership,
            )


async def test_audit_record_for_one_tenant_is_invisible_to_another(app_settings, database_urls):
    """AC2: an audit record written for one tenant is invisible to a second tenant's session
    under the standard RLS policy."""
    tenant_a = await _seed_tenant(database_urls["superuser"])
    tenant_b = await _seed_tenant(database_urls["superuser"])
    _, member_a = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_a, role="member"
    )

    ctx_a = RequestContext(
        tenant_id=tenant_a, identity_id=uuid.uuid4(), roles=frozenset({"member"})
    )
    repo = ApprovalAuditRepository()
    pending_action_id = uuid.uuid4()
    async with tenant_session(ctx_a) as session:
        event = await repo.record(
            session,
            ctx_a,
            kind=REQUESTED,
            tool_name="send_invoice",
            actor_membership_id=member_a,
            pending_action_id=pending_action_id,
        )
        event_id = event.id

    ctx_b = RequestContext(
        tenant_id=tenant_b, identity_id=uuid.uuid4(), roles=frozenset({"member"})
    )
    async with tenant_session(ctx_b) as session:
        listing_b = await repo.list_for_pending_action(
            session, ctx_b, pending_action_id=pending_action_id
        )
        assert listing_b == []

        # Directly against the table, bypassing the repository's own WHERE clause entirely.
        row = (
            await session.execute(
                text("SELECT id FROM approval_audit_events WHERE id = :id"), {"id": event_id}
            )
        ).first()
        assert row is None

    # Still visible, under its own tenant.
    async with tenant_session(ctx_a) as session:
        listing_a = await repo.list_for_pending_action(
            session, ctx_a, pending_action_id=pending_action_id
        )
        assert [record.id for record in listing_a] == [event_id]


async def test_app_role_has_no_update_or_delete_grant_on_the_audit_table(
    app_settings, database_urls
):
    """AC3: the repository exposes no update/delete method (see the module -- `record()`,
    `list_for_pending_action()`, `list_for_standing_grant()` and nothing else), and the `app`
    role itself cannot update or delete a row even bypassing the repository entirely --
    append-only is a property of what can be called, not a convention to remember."""
    assert not hasattr(ApprovalAuditRepository, "update")
    assert not hasattr(ApprovalAuditRepository, "delete")

    tenant_id = await _seed_tenant(database_urls["superuser"])
    _, member_membership = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="member"
    )
    ctx = RequestContext(tenant_id=tenant_id, identity_id=uuid.uuid4(), roles=frozenset({"member"}))
    repo = ApprovalAuditRepository()

    async with tenant_session(ctx) as session:
        event = await repo.record(
            session,
            ctx,
            kind=REQUESTED,
            tool_name="send_invoice",
            actor_membership_id=member_membership,
        )
        event_id = event.id

    app_engine = create_async_engine(database_urls["app"])
    async with app_engine.connect() as conn:
        async with conn.begin():
            await conn.execute(text(f"SET LOCAL app.tenant_id = '{tenant_id}'"))
            with pytest.raises(DBAPIError, match="permission denied"):
                await conn.execute(
                    text("UPDATE approval_audit_events SET kind = 'approved' WHERE id = :id"),
                    {"id": event_id},
                )

    async with app_engine.connect() as conn:
        async with conn.begin():
            await conn.execute(text(f"SET LOCAL app.tenant_id = '{tenant_id}'"))
            with pytest.raises(DBAPIError, match="permission denied"):
                await conn.execute(
                    text("DELETE FROM approval_audit_events WHERE id = :id"), {"id": event_id}
                )
    await app_engine.dispose()


async def test_read_for_one_pending_action_orders_most_recent_first(app_settings, database_urls):
    """AC4: a tenant-scoped read returns the milestones for one pending action in an order that
    makes the most recent one easy to find (most recent first)."""
    tenant_id = await _seed_tenant(database_urls["superuser"])
    _, member_membership = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="member"
    )
    ctx = RequestContext(tenant_id=tenant_id, identity_id=uuid.uuid4(), roles=frozenset({"member"}))
    repo = ApprovalAuditRepository()
    pending_action_id = uuid.uuid4()

    async with tenant_session(ctx) as session:
        first = await repo.record(
            session,
            ctx,
            kind=REQUESTED,
            tool_name="send_invoice",
            actor_membership_id=member_membership,
            pending_action_id=pending_action_id,
        )
        second = await repo.record(
            session,
            ctx,
            kind=APPROVED,
            tool_name="send_invoice",
            actor_membership_id=member_membership,
            pending_action_id=pending_action_id,
        )
        third = await repo.record(
            session,
            ctx,
            kind=EXECUTED,
            tool_name="send_invoice",
            actor_membership_id=member_membership,
            pending_action_id=pending_action_id,
        )

        listing = await repo.list_for_pending_action(
            session, ctx, pending_action_id=pending_action_id
        )

    assert [record.id for record in listing] == [third.id, second.id, first.id]


async def test_read_for_one_standing_grant_orders_most_recent_first(app_settings, database_urls):
    """The equivalent read for a standing grant's own milestones, most recent first."""
    tenant_id = await _seed_tenant(database_urls["superuser"])
    _, agent_membership = await _seed_membership(
        database_urls["superuser"], tenant_id=tenant_id, role="agent"
    )
    ctx = RequestContext(tenant_id=tenant_id, identity_id=uuid.uuid4(), roles=frozenset({"agent"}))
    repo = ApprovalAuditRepository()
    standing_grant_id = uuid.uuid4()

    async with tenant_session(ctx) as session:
        first = await repo.record(
            session,
            ctx,
            kind=EXECUTED,
            tool_name="send_invoice",
            actor_membership_id=agent_membership,
            standing_grant_id=standing_grant_id,
        )
        second = await repo.record(
            session,
            ctx,
            kind=FAILED_TO_EXECUTE,
            tool_name="send_invoice",
            actor_membership_id=agent_membership,
            standing_grant_id=standing_grant_id,
        )

        listing = await repo.list_for_standing_grant(
            session, ctx, standing_grant_id=standing_grant_id
        )

    assert [record.id for record in listing] == [second.id, first.id]
