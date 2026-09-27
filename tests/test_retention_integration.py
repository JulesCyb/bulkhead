"""Real isolation test for the conversation retention job (ADR-0006, Spec 4 / #35) against
PostgreSQL + pgvector (`pgserver`, via `uv sync --group dbtest`). Pattern:
`tests/test_standing_grants_integration.py`.

Proves what a unit test on `ConversationsRepository.delete_expired` alone cannot (that contract
is already covered by `tests/test_rls_integration.py`): that `app/retention.py`'s job actually
visits every tenant the control plane knows about, applies each tenant's *own* retention cutoff
(its own `settings["retention_days"]`, or the documented default), deletes only what that one
tenant's own transaction is allowed to touch, and never issues a single statement against
`conversations`/`messages` on the connection it uses only to enumerate tenants.

Suspended tenants are skipped by the job (#106, ADR-0010); the last test below proves it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import create_async_engine

from app.tenant_settings import DEFAULT_RETENTION_DAYS  # noqa: E402

pgserver = pytest.importorskip("pgserver")

import scripts.retention as retention_script  # noqa: E402
from app.retention import run_retention_job  # noqa: E402
from tests.support import (  # noqa: E402
    cluster,
    environment,
    seed_conversation,
    seed_membership,
    seed_tenant,
)
from tests.support.seeding import set_tenant_retention_days  # noqa: E402

_ = (cluster, environment)


async def _conversation_ids(url: str, tenant_id) -> list[str]:
    engine = create_async_engine(url)
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    text(
                        "SELECT conversation_id FROM conversations WHERE tenant_id = :tid "
                        "ORDER BY conversation_id"
                    ),
                    {"tid": tenant_id},
                )
            )
            .scalars()
            .all()
        )
    await engine.dispose()
    return list(rows)


async def _message_count(url: str, *, tenant_id, conversation_id: str) -> int:
    engine = create_async_engine(url)
    async with engine.connect() as conn:
        count = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM messages "
                    "WHERE tenant_id = :tid AND conversation_id = :cid"
                ),
                {"tid": tenant_id, "cid": conversation_id},
            )
        ).scalar_one()
    await engine.dispose()
    return count


class _FakeSecret:
    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value


class _FakeMigrationSettings:
    def __init__(self, url: str) -> None:
        self.database_url_migrations = _FakeSecret(url)


async def test_job_deletes_only_each_tenants_own_expired_conversations(environment):
    """AC1: sweeping every tenant in one job run deletes only the currently-processed tenant's
    conversations whose last activity is past its own cutoff -- that tenant's fresher
    conversations, and another tenant's conversations, are left untouched."""
    now = datetime.now(UTC)
    tenant_a = await seed_tenant(environment, name="Acme", via_operator=False)
    tenant_b = await seed_tenant(environment, name="Globex", via_operator=False)
    identity_a, _ = await seed_membership(environment, tenant_id=tenant_a.tenant_id, role="member")

    # Well past the documented default (90 days) -- expired for tenant A.
    await seed_conversation(
        environment,
        tenant_id=tenant_a.tenant_id,
        identity_id=identity_a,
        conversation_id="conv-old",
        last_activity_at=now - timedelta(days=DEFAULT_RETENTION_DAYS + 30),
        with_message=True,
    )
    # Recent -- not expired for tenant A.
    await seed_conversation(
        environment,
        tenant_id=tenant_a.tenant_id,
        identity_id=identity_a,
        conversation_id="conv-fresh",
        last_activity_at=now,
        with_message=True,
    )
    # Tenant B's own conversation is recent too -- must survive the sweep untouched even though
    # the job visits tenant B in the very same run.
    identity_b, _ = await seed_membership(environment, tenant_id=tenant_b.tenant_id, role="member")
    await seed_conversation(
        environment,
        tenant_id=tenant_b.tenant_id,
        identity_id=identity_b,
        conversation_id="conv-b",
        last_activity_at=now,
        with_message=True,
    )

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            outcomes = await run_retention_job(conn)
    finally:
        await engine.dispose()

    by_tenant = {outcome.tenant_id: outcome.deleted for outcome in outcomes}
    assert by_tenant[tenant_a.tenant_id] == 1
    assert by_tenant[tenant_b.tenant_id] == 0

    assert await _conversation_ids(environment.superuser_url, tenant_a.tenant_id) == ["conv-fresh"]
    assert await _conversation_ids(environment.superuser_url, tenant_b.tenant_id) == ["conv-b"]


async def test_a_tenants_own_retention_setting_is_honored_over_the_default(environment):
    """AC2: a tenant that has set its own (shorter) retention period on its tenant settings is
    expired using that period as the cutoff, not the documented default -- proving the setting is
    actually honored, not just readable. 20 days old is *inside* the 90-day default window (would
    survive under the default) but past this tenant's own 10-day setting."""
    now = datetime.now(UTC)
    tenant = await seed_tenant(environment, name="ShortRetention", via_operator=False)
    await set_tenant_retention_days(environment, tenant.tenant_id, 10)

    identity, _ = await seed_membership(environment, tenant_id=tenant.tenant_id, role="member")
    await seed_conversation(
        environment,
        tenant_id=tenant.tenant_id,
        identity_id=identity,
        conversation_id="conv-20-days-old",
        last_activity_at=now - timedelta(days=20),
        with_message=True,
    )

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            outcomes = await run_retention_job(conn)
    finally:
        await engine.dispose()

    assert {o.tenant_id: o.deleted for o in outcomes}[tenant.tenant_id] == 1
    assert await _conversation_ids(environment.superuser_url, tenant.tenant_id) == []


async def test_job_leaves_no_orphaned_messages_behind(environment):
    """AC3: deleting an expired conversation removes its messages with it -- no orphaned message
    rows survive the sweep."""
    now = datetime.now(UTC)

    tenant = await seed_tenant(environment, name="Initech", via_operator=False)
    identity, _ = await seed_membership(environment, tenant_id=tenant.tenant_id, role="member")
    await seed_conversation(
        environment,
        tenant_id=tenant.tenant_id,
        identity_id=identity,
        conversation_id="conv-expired",
        last_activity_at=now - timedelta(days=DEFAULT_RETENTION_DAYS + 1),
        with_message=True,
    )

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            await run_retention_job(conn)
    finally:
        await engine.dispose()

    count = await _message_count(
        environment.superuser_url, tenant_id=tenant.tenant_id, conversation_id="conv-expired"
    )
    assert count == 0


async def test_job_never_touches_conversations_or_messages_on_the_enumeration_connection(
    environment,
):
    """AC4: the connection passed to `run_retention_job` (an `app_owner` connection) is used only
    to enumerate tenants -- every statement it issues names `control.enumerate_tenants()` and
    nothing else. The per-tenant deletion always happens on a separate connection, opened by
    `tenant_session(ctx)` (the ordinary, RLS-scoped `app`-role path)."""
    now = datetime.now(UTC)

    tenant = await seed_tenant(environment, name="Umbrella", via_operator=False)
    identity, _ = await seed_membership(environment, tenant_id=tenant.tenant_id, role="member")
    await seed_conversation(
        environment,
        tenant_id=tenant.tenant_id,
        identity_id=identity,
        conversation_id="conv-expired",
        last_activity_at=now - timedelta(days=DEFAULT_RETENTION_DAYS + 1),
        with_message=True,
    )

    engine = create_async_engine(environment.owner_url)
    statements: list[str] = []

    def _capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", _capture)
    try:
        async with engine.begin() as conn:
            await run_retention_job(conn)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", _capture)
        await engine.dispose()

    assert statements, "expected at least the enumerate_tenants() call to be captured"
    for statement in statements:
        lowered = statement.lower()
        assert "conversations" not in lowered
        assert "messages" not in lowered
    assert any("enumerate_tenants" in statement.lower() for statement in statements)

    # And the deletion really did happen -- just not on this connection.
    assert await _conversation_ids(environment.superuser_url, tenant.tenant_id) == []


async def test_retention_script_entry_point_runs_end_to_end(environment, monkeypatch):
    """The documented, runnable script (`uv run python scripts/retention.py`) actually deletes
    expired conversations end to end, through its real async entry point (`_main`, which `main()`
    only wraps in `asyncio.run` -- called directly here since this test already runs inside an
    event loop) -- only `get_migration_settings` is faked, to point it at the ephemeral test
    database instead of reading `.env`/the process environment for a real owner DSN."""
    now = datetime.now(UTC)

    tenant = await seed_tenant(environment, name="Soylent", via_operator=False)
    identity, _ = await seed_membership(environment, tenant_id=tenant.tenant_id, role="member")
    await seed_conversation(
        environment,
        tenant_id=tenant.tenant_id,
        identity_id=identity,
        conversation_id="conv-expired",
        last_activity_at=now - timedelta(days=DEFAULT_RETENTION_DAYS + 1),
        with_message=True,
    )
    await seed_conversation(
        environment,
        tenant_id=tenant.tenant_id,
        identity_id=identity,
        conversation_id="conv-fresh",
        last_activity_at=now,
        with_message=True,
    )

    monkeypatch.setattr(
        retention_script,
        "get_migration_settings",
        lambda: _FakeMigrationSettings(environment.owner_url),
    )

    await retention_script._main()

    assert await _conversation_ids(environment.superuser_url, tenant.tenant_id) == ["conv-fresh"]


async def test_a_suspended_tenants_conversations_are_untouched_by_the_job(environment):
    """#106, ADR-0010: a suspended tenant is skipped outright (one log line, no `tenant_session()`
    opened for it) rather than raising `TenantSuspendedError` mid-sweep -- its own expired
    conversation survives the run untouched, and it is not counted among the returned outcomes,
    while an unsuspended tenant in the very same run is swept normally."""
    now = datetime.now(UTC)
    suspended = await seed_tenant(environment, name="Suspended", via_operator=False)
    active = await seed_tenant(environment, name="Active", via_operator=False)
    identity_suspended, _ = await seed_membership(
        environment, tenant_id=suspended.tenant_id, role="member"
    )
    identity_active, _ = await seed_membership(
        environment, tenant_id=active.tenant_id, role="member"
    )
    await seed_conversation(
        environment,
        tenant_id=suspended.tenant_id,
        identity_id=identity_suspended,
        conversation_id="conv-suspended-expired",
        last_activity_at=now - timedelta(days=DEFAULT_RETENTION_DAYS + 30),
    )
    await seed_conversation(
        environment,
        tenant_id=active.tenant_id,
        identity_id=identity_active,
        conversation_id="conv-active-expired",
        last_activity_at=now - timedelta(days=DEFAULT_RETENTION_DAYS + 30),
    )
    await suspended.suspend()

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            outcomes = await run_retention_job(conn)
    finally:
        await engine.dispose()

    assert suspended.tenant_id not in {o.tenant_id for o in outcomes}
    assert {o.tenant_id: o.deleted for o in outcomes}[active.tenant_id] == 1

    assert await _conversation_ids(environment.superuser_url, suspended.tenant_id) == [
        "conv-suspended-expired"
    ]
    assert await _conversation_ids(environment.superuser_url, active.tenant_id) == []


async def test_a_stored_retention_days_far_above_the_maximum_is_clamped_by_the_job(
    environment, caplog
):
    """#84 (ADR-0006; GDPR Art. 5(1)(e), storage limitation): a tenant whose stored
    `retention_days` is far above the deployment maximum (simulating a row written under an
    earlier, higher cap -- or before the cap existed at all, since no write path could produce
    this value today) has its cutoff clamped to that maximum by `effective_retention_days`, not
    honored as-is -- a conversation older than the maximum but nowhere near the enormous stored
    value is deleted anyway, proving the clamp (not the tenant's own setting) governs. A second
    tenant whose stored value is legitimately below the maximum is completely untouched by the
    clamp -- its own (still-honored) setting keeps governing exactly as before (#84 does not
    change the already-covered "own setting wins over the default" behavior)."""
    from app.config import Settings

    now = datetime.now(UTC)
    max_days = 100

    over_cap = await seed_tenant(environment, name="OverCap", via_operator=False)
    await set_tenant_retention_days(environment, over_cap.tenant_id, 10_000)
    identity_over, _ = await seed_membership(
        environment, tenant_id=over_cap.tenant_id, role="member"
    )
    # Older than max_days (100) but nowhere near the tenant's own stored 10,000-day setting --
    # expired only once the clamp actually applies.
    await seed_conversation(
        environment,
        tenant_id=over_cap.tenant_id,
        identity_id=identity_over,
        conversation_id="conv-over-cap-expired",
        last_activity_at=now - timedelta(days=max_days + 10),
        with_message=True,
    )

    under_cap = await seed_tenant(environment, name="UnderCap", via_operator=False)
    await set_tenant_retention_days(environment, under_cap.tenant_id, 30)
    identity_under, _ = await seed_membership(
        environment, tenant_id=under_cap.tenant_id, role="member"
    )
    # 20 days old: inside its own 30-day setting, and well inside max_days too -- must survive.
    await seed_conversation(
        environment,
        tenant_id=under_cap.tenant_id,
        identity_id=identity_under,
        conversation_id="conv-under-cap-fresh",
        last_activity_at=now - timedelta(days=20),
        with_message=True,
    )

    settings = Settings(max_retention_days=max_days)
    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            with caplog.at_level("WARNING"):
                outcomes = await run_retention_job(conn, settings=settings)
    finally:
        await engine.dispose()

    by_tenant = {o.tenant_id: o.deleted for o in outcomes}
    assert by_tenant[over_cap.tenant_id] == 1
    assert by_tenant[under_cap.tenant_id] == 0

    assert await _conversation_ids(environment.superuser_url, over_cap.tenant_id) == []
    assert await _conversation_ids(environment.superuser_url, under_cap.tenant_id) == [
        "conv-under-cap-fresh"
    ]

    # One log line names the clamped tenant and its stored value; the under-cap tenant (never
    # clamped) is never mentioned by it.
    assert caplog.text.count("clamping") == 1
    assert str(over_cap.tenant_id) in caplog.text
    assert "10000" in caplog.text
    assert str(under_cap.tenant_id) not in caplog.text
