"""Tests for the test support package itself (issue #96 / spec #90, "A6").

Proves the two guarantees every later integration test in this suite leans on: that
`seed_tenant` really isolates two tenants' documents and memberships through the real session
layer (not just by construction), and that its NOT-NULL-column guard really fires when a table
grows a column the seeder does not know about, instead of failing with a cryptic constraint
violation or silently inserting a wrong default.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

pgserver = pytest.importorskip("pgserver")

from app.db.session import tenant_session  # noqa: E402
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
