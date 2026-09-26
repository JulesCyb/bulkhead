"""Seeding a pooled tenant: `tenants`, `control.tenants`, `control.identities`, `memberships`,
and `documents` -- the raw statements every one of the (formerly thirty-three) hand-written seed
helpers duplicated (issue #96 / spec #90, "A6"). Dedicated-tier seeding is out of scope for this
ticket (ADR-0002's isolation tier stays `'pooled'` here, per `SeededTenant.isolation_tier`); a
later ticket extends this module for it.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from app.context import MeansKind, RequestContext, Role
from tests.support.cluster import Cluster

DIM = 1536


def vector_literal(seed: float) -> str:
    """A deterministic 1536-dim pgvector literal, distinct per `seed` -- the same shape
    `tests/test_rls_integration.py`'s own `_vec` helper produced, now shared."""
    values = [0.0] * DIM
    values[0] = 1.0
    values[1] = seed
    return "[" + ",".join(f"{v:.3f}" for v in values) + "]"


# Every column this module's INSERTs set explicitly, per table -- compared against a live
# NOT-NULL-without-default catalog read in `assert_known_not_null_columns` below. A table
# gaining such a column that isn't listed here fails seeding loudly instead of the insert
# quietly succeeding, or failing with a bare NOT NULL constraint violation that names the
# symptom but not "the seeder needs to learn about this column".
_KNOWN_NOT_NULL_COLUMNS: dict[str, frozenset[str]] = {
    "public.tenants": frozenset({"id", "name"}),
    "control.tenants": frozenset({"tenant_id", "residency", "isolation_tier"}),
    "control.identities": frozenset({"id", "issuer", "subject", "kind"}),
    "public.memberships": frozenset({"id", "tenant_id", "identity_id", "role"}),
    "public.documents": frozenset({"id", "tenant_id", "title", "content", "embedding"}),
}


class UnknownNotNullColumnError(RuntimeError):
    """Raised when a table this module seeds has grown a NOT NULL column with no default that
    isn't in `_KNOWN_NOT_NULL_COLUMNS` (or a caller's own mapping passed directly to
    `assert_known_not_null_columns`) -- schema drift the seeder does not yet know how to fill,
    surfaced here with the table and column named, instead of as a bare constraint violation
    from deep inside an INSERT."""

    def __init__(self, missing: Mapping[str, set[str]]) -> None:
        self.missing = {table: set(cols) for table, cols in missing.items()}
        detail = ", ".join(f"{table}: {sorted(cols)}" for table, cols in sorted(missing.items()))
        super().__init__(
            "seeding does not know about the following NOT NULL column(s) without a default -- "
            f"{detail}"
        )


async def assert_known_not_null_columns(
    conn: AsyncConnection, known: Mapping[str, frozenset[str]]
) -> None:
    """Reads `information_schema.columns` once for every table `known` names and raises
    `UnknownNotNullColumnError` if any of them has a NOT NULL, no-default column that isn't in
    its known set.

    A key may be schema-qualified (`"control.tenants"`) or bare (`"documents"`, or a temp
    table's own name) -- a bare key matches a table of that name in any schema, which is what
    lets a test point this same check at a throwaway temp table without needing to know its
    (session-specific) temp schema name.
    """
    names = sorted({key.rsplit(".", 1)[-1] for key in known})
    rows = (
        await conn.execute(
            text(
                "SELECT table_schema, table_name, column_name FROM information_schema.columns "
                "WHERE is_nullable = 'NO' AND column_default IS NULL "
                "AND table_name = ANY(:names)"
            ),
            {"names": names},
        )
    ).all()

    missing: dict[str, set[str]] = defaultdict(set)
    for schema, table, column in rows:
        qualified = f"{schema}.{table}"
        key = qualified if qualified in known else (table if table in known else None)
        if key is None:
            continue
        if column not in known[key]:
            missing[qualified].add(column)

    if missing:
        raise UnknownNotNullColumnError(missing)


@dataclass(frozen=True, slots=True)
class SeededTenant:
    """Everything a test needs about one seeded pooled tenant: real `tenants`/`control.tenants`
    rows, one real `control.identities` + `memberships` row per role `seed_tenant` was asked
    for, and any seeded `documents`."""

    tenant_id: uuid.UUID
    name: str
    residency: str | None
    isolation_tier: str
    identities: dict[Role, uuid.UUID]
    memberships: dict[Role, uuid.UUID]
    document_ids: list[uuid.UUID]
    cluster: Cluster

    def ctx(self, role: Role, *, means: tuple[MeansKind, str] | None = None) -> RequestContext:
        """A `RequestContext` for the identity seeded under `role` -- the tenant, that identity,
        and that one role, exactly as a real request for that membership would resolve to.
        `means=(kind, id)` builds it through `RequestContext.acting_through` instead, for a test
        that needs a delegation/credential means attached (ADR-0005)."""
        ctx = RequestContext(
            tenant_id=self.tenant_id, identity_id=self.identities[role], roles=frozenset({role})
        )
        if means is None:
            return ctx
        return ctx.acting_through(*means)

    @asynccontextmanager
    async def owner_connection(self) -> AsyncIterator[AsyncConnection]:
        """A connection to this tenant's database as the owner role (`app_owner`) -- for
        planting an edge case or reading a row back directly, bypassing the repository layer."""
        async with _connection(self.cluster.owner_url) as conn:
            yield conn

    @asynccontextmanager
    async def superuser_connection(self) -> AsyncIterator[AsyncConnection]:
        """A connection as the cluster's own bootstrap superuser (test-only -- see `Cluster`)."""
        async with _connection(self.cluster.superuser_url) as conn:
            yield conn


@asynccontextmanager
async def _connection(url: str) -> AsyncIterator[AsyncConnection]:
    engine: AsyncEngine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            yield conn
    finally:
        await engine.dispose()


async def seed_membership(
    cluster: Cluster, *, tenant_id: uuid.UUID, role: Role
) -> tuple[uuid.UUID, uuid.UUID]:
    """A global identity plus its membership of `role` in `tenant_id`, as the owner role.
    Returns `(identity_id, membership_id)`. The seam a test reaches for when it needs a
    membership `seed_tenant`'s own `roles=` argument does not cover -- a second membership of a
    role already requested, for example. Writes as the cluster's own superuser -- same reason as
    `seed_tenant` above."""
    engine = create_async_engine(cluster.superuser_url)
    identity_id = uuid.uuid4()
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO control.identities (id, issuer, subject) "
                    "VALUES (:id, 'seed', :sub)"
                ),
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
    finally:
        await engine.dispose()
    return identity_id, membership_id


async def seed_tenant(
    cluster: Cluster,
    *,
    name: str = "Acme",
    residency: str | None = None,
    roles: Iterable[Role] = (),
    documents: int = 0,
) -> SeededTenant:
    """Seeds one pooled tenant (ADR-0002 -- dedicated-tier seeding is later ticket work): a
    `tenants` row, its `control.tenants` row (residency, isolation tier), one real
    `control.identities` + `memberships` row per role in `roles`, and `documents` documents with
    a 1536-dim embedding, attributed to the first seeded role (or a dedicated seed identity if
    `roles` is empty).

    Writes as the cluster's own superuser (test-only, like every other seed helper this package
    replaces -- see `Cluster`): `app_owner` does not bypass RLS either, so it cannot insert a
    brand-new tenant row without `app.tenant_id` already set to that row's own id, exactly the
    chicken-and-egg problem plain test seeding (unlike a real request, which never creates its
    own tenant row) always runs into. `app.identity_id` is still set explicitly before the
    document inserts below, exactly like a real `tenant_session()` write would set it, so
    `documents.created_by`/`updated_by` (migration 0010) resolves the same way.
    """
    engine = create_async_engine(cluster.superuser_url)
    tenant_id = uuid.uuid4()
    seed_identity_id = uuid.uuid4()
    identities: dict[Role, uuid.UUID] = {}
    memberships: dict[Role, uuid.UUID] = {}
    document_ids: list[uuid.UUID] = []

    try:
        async with engine.begin() as conn:
            await assert_known_not_null_columns(conn, _KNOWN_NOT_NULL_COLUMNS)

            await conn.execute(
                text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
                {"id": tenant_id, "name": name},
            )

            control_fields: dict[str, object] = {"tenant_id": tenant_id}
            if residency is not None:
                control_fields["residency"] = residency
            columns = ", ".join(control_fields)
            placeholders = ", ".join(f":{c}" for c in control_fields)
            await conn.execute(
                text(f"INSERT INTO control.tenants ({columns}) VALUES ({placeholders})"),
                control_fields,
            )

            await conn.execute(
                text(
                    "INSERT INTO control.identities (id, issuer, subject) "
                    "VALUES (:id, 'seed', :sub)"
                ),
                {"id": seed_identity_id, "sub": str(seed_identity_id)},
            )

            for role in roles:
                identity_id = uuid.uuid4()
                await conn.execute(
                    text(
                        "INSERT INTO control.identities (id, issuer, subject) "
                        "VALUES (:id, 'seed', :sub)"
                    ),
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
                identities[role] = identity_id
                memberships[role] = membership_id

            if documents:
                creator_id = next(iter(identities.values()), seed_identity_id)
                await conn.execute(
                    text("SELECT set_config('app.identity_id', :iid, true)"),
                    {"iid": str(creator_id)},
                )
                for index in range(documents):
                    doc_id = (
                        await conn.execute(
                            text(
                                "INSERT INTO documents (tenant_id, title, content, embedding) "
                                "VALUES (:tid, :title, :content, CAST(:emb AS vector)) "
                                "RETURNING id"
                            ),
                            {
                                "tid": tenant_id,
                                "title": f"Document {index}",
                                "content": f"Content {index}",
                                "emb": vector_literal(0.1 * (index + 1)),
                            },
                        )
                    ).scalar_one()
                    document_ids.append(doc_id)
    finally:
        await engine.dispose()

    return SeededTenant(
        tenant_id=tenant_id,
        name=name,
        residency=residency,
        isolation_tier="pooled",
        identities=identities,
        memberships=memberships,
        document_ids=document_ids,
        cluster=cluster,
    )
