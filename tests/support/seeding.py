"""Seeding a tenant, pooled or dedicated: `tenants`, `control.tenants`, `control.identities`,
`memberships`, and `documents` -- the raw statements every one of the (formerly thirty-three)
hand-written seed helpers duplicated (issue #96 / spec #90, "A6"; dedicated-tier seeding is
issue #97 / "A6-T2").

A pooled tenant's rows all live in the one pooled database `cluster` (the argument every
`seed_tenant` call takes) already points at. A dedicated tenant's own rows (`tenants`,
`memberships`, `documents`) live in its own, separate database instead -- created on the same
embedded cluster, migrated to head with the real runner, exactly as ADR-0002 describes -- while
its control-plane bookkeeping (`control.tenants`: residency, isolation tier, database alias)
stays in the pooled database, the one place the control plane ever lives.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

import scripts.migrate as migrate_module
from app.context import MeansKind, RequestContext, Role
from app.operator.dedicated_db import generate_database_alias
from tests.support.cluster import (
    Cluster,
    create_database,
    migration_run_without_disrupting_logging,
)

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
    """Everything a test needs about one seeded tenant, pooled or dedicated: real
    `tenants`/`control.tenants` rows, one real `control.identities` + `memberships` row per role
    `seed_tenant` was asked for, and any seeded `documents`.

    `cluster` is always the pooled cluster -- the control plane's own home, regardless of tier.
    `database` is `None` for a pooled tenant (its data lives in `cluster` too) or the tenant's
    own dedicated `Cluster` (`database_alias`, `owner_url`, `app_url` all describe it) when
    `isolation_tier == "dedicated"`; `owner_connection`/`superuser_connection` below always reach
    wherever the tenant's *own* rows actually live, not the pooled cluster, once it has one.
    """

    tenant_id: uuid.UUID
    name: str
    residency: str | None
    isolation_tier: str
    identities: dict[Role, uuid.UUID]
    memberships: dict[Role, uuid.UUID]
    document_ids: list[uuid.UUID]
    cluster: Cluster
    database_alias: str | None = None
    database: Cluster | None = None

    @property
    def owner_url(self) -> str:
        """The owner-role (`app_owner`) DSN for this tenant's own database -- the pooled
        cluster's for a pooled tenant, or the dedicated database's for a dedicated one."""
        return (self.database or self.cluster).owner_url

    @property
    def app_url(self) -> str:
        """The app-role (`app`) DSN for this tenant's own database -- same rule as `owner_url`."""
        return (self.database or self.cluster).app_url

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
        """A connection to this tenant's own database as the owner role (`app_owner`) -- for
        planting an edge case or reading a row back directly, bypassing the repository layer."""
        async with _connection(self.owner_url) as conn:
            yield conn

    @asynccontextmanager
    async def superuser_connection(self) -> AsyncIterator[AsyncConnection]:
        """A connection to this tenant's own database as its bootstrap superuser (test-only --
        see `Cluster`)."""
        async with _connection((self.database or self.cluster).superuser_url) as conn:
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


def _require_secrets_dir(env_var: str) -> Path:
    """The temporary directory `env_var` currently points at -- raises instead of falling back
    to that variable's production default (`app/db/engine_registry.py`'s
    `/run/secrets/tenant-db`, `scripts/migrate.py`'s `/run/secrets/tenant-db-migrations`) when
    it is unset. Dedicated-tier seeding requires the `environment` fixture (`tests.support.
    cluster`), which sets both -- this is the fail-closed guard against a test that used the
    bare `cluster` fixture instead and would otherwise silently read or write a real secrets
    path."""
    value = os.environ.get(env_var)
    if not value:
        raise RuntimeError(
            f"seed_tenant(isolation_tier='dedicated') requires {env_var} to already point at a "
            "temporary directory -- use the `environment` fixture (tests.support.cluster), not "
            "the bare `cluster` fixture, so this never falls back to reading or writing that "
            "variable's production secrets path."
        )
    return Path(value)


async def _provision_dedicated_tenant_database(cluster: Cluster, alias: str) -> Cluster:
    """Creates `alias`'s own database on `cluster`'s server (`tests.support.cluster.
    create_database`), migrates it to head with the real runner -- `scripts.migrate.
    migrate_alias`, the exact code path production uses for a dedicated alias -- and writes both
    its owner-role and app-role tenant-secret files where `scripts/migrate.py` and
    `app/db/engine_registry.py` read them respectively, so a test that later routes through
    `tenant_session(ctx)` or runs `scripts/migrate.py` against this alias needs no seam of its
    own."""
    # Both directories checked upfront, before any database is created: a missing one must fail
    # before any side effect, not after `CREATE DATABASE` has already run.
    migrations_dir = _require_secrets_dir("TENANT_DB_MIGRATIONS_SECRETS_DIR")
    app_dir = _require_secrets_dir("TENANT_DB_SECRETS_DIR")

    dedicated = await create_database(cluster, alias)

    migrations_secret = migrations_dir / alias
    migrations_secret.parent.mkdir(parents=True, exist_ok=True)
    migrations_secret.write_text(dedicated.owner_url)

    # scripts.migrate.migrate_alias calls alembic's command.upgrade(), which (migrations/env.py)
    # itself calls asyncio.run() -- fatal if invoked directly from a coroutine already running
    # inside an event loop (this one), so it runs in a worker thread instead, exactly like
    # app.operator.dedicated_db.ensure_dedicated_database does for the same reason.
    with migration_run_without_disrupting_logging():
        await asyncio.to_thread(migrate_module.migrate_alias, alias)

    app_secret = app_dir / alias
    app_secret.parent.mkdir(parents=True, exist_ok=True)
    app_secret.write_text(dedicated.app_url)

    return dedicated


async def _insert_tenant_data_rows(
    conn: AsyncConnection,
    *,
    tenant_id: uuid.UUID,
    name: str,
    seed_identity_id: uuid.UUID,
    roles: Iterable[Role],
    documents: int,
    mirror_tenant_row: bool,
) -> tuple[dict[Role, uuid.UUID], dict[Role, uuid.UUID], list[uuid.UUID]]:
    """The tenant's own rows -- optionally its `tenants` row (`mirror_tenant_row`: a pooled
    tenant's row is inserted by its caller, in the same transaction, before this runs; a
    dedicated tenant's own database needs its own copy, since `memberships`/`documents` there
    foreign-key to it locally), one `control.identities` + `memberships` row per role in
    `roles`, and `documents` documents with a 1536-dim embedding, attributed to the first seeded
    role (or a dedicated seed identity if `roles` is empty). `app.identity_id` is set explicitly
    before the document inserts, exactly like a real `tenant_session()` write would set it, so
    `documents.created_by`/`updated_by` (migration 0010) resolves the same way. Shared by both of
    `seed_tenant`'s tiers -- see its own docstring for which connection/transaction each uses."""
    identities: dict[Role, uuid.UUID] = {}
    memberships: dict[Role, uuid.UUID] = {}
    document_ids: list[uuid.UUID] = []

    if mirror_tenant_row:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant_id, "name": name},
        )

    await conn.execute(
        text("INSERT INTO control.identities (id, issuer, subject) VALUES (:id, 'seed', :sub)"),
        {"id": seed_identity_id, "sub": str(seed_identity_id)},
    )

    for role in roles:
        identity_id = uuid.uuid4()
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

    return identities, memberships, document_ids


async def seed_tenant(
    cluster: Cluster,
    *,
    name: str = "Acme",
    residency: str | None = None,
    roles: Iterable[Role] = (),
    documents: int = 0,
    isolation_tier: str = "pooled",
) -> SeededTenant:
    """Seeds one tenant (ADR-0002): a `tenants` row, its `control.tenants` row (residency,
    isolation tier, and -- for a dedicated tenant -- its database alias), one real
    `control.identities` + `memberships` row per role in `roles`, and `documents` documents with
    a 1536-dim embedding.

    `isolation_tier="pooled"` (the default) writes every row into `cluster` itself, in one
    transaction, as the cluster's own superuser (test-only, like every other seed helper this
    package replaces -- see `Cluster`): `app_owner` does not bypass RLS either, so it cannot
    insert a brand-new tenant row without `app.tenant_id` already set to that row's own id,
    exactly the chicken-and-egg problem plain test seeding (unlike a real request, which never
    creates its own tenant row) always runs into.

    `isolation_tier="dedicated"` additionally creates the tenant's own database on `cluster`'s
    server, migrates it to head with the real runner, and writes its owner-role/app-role secret
    files where `scripts/migrate.py`/`app/db/engine_registry.py` read them (see
    `_provision_dedicated_tenant_database`; requires the `environment` fixture, which points
    both secrets directories at temporary ones). Its `control.tenants` bookkeeping row (residency,
    isolation tier, database alias) still lives in the pooled `cluster` -- the control plane is
    never itself sharded -- but its own `tenants`/`memberships`/`documents` rows are written into
    its own dedicated database instead. The returned `SeededTenant.database_alias`/`owner_url`/
    `app_url` describe that database; `SeededTenant.ctx()` is unchanged either way -- a real
    request never knows or cares which database serves it.
    """
    if isolation_tier not in ("pooled", "dedicated"):
        raise ValueError(f"seed_tenant: unknown isolation_tier {isolation_tier!r}")

    tenant_id = uuid.uuid4()
    seed_identity_id = uuid.uuid4()
    identities: dict[Role, uuid.UUID] = {}
    memberships: dict[Role, uuid.UUID] = {}
    document_ids: list[uuid.UUID] = []

    database_alias: str | None = None
    dedicated: Cluster | None = None
    if isolation_tier == "dedicated":
        database_alias = generate_database_alias(tenant_id)
        dedicated = await _provision_dedicated_tenant_database(cluster, database_alias)

    pooled_engine = create_async_engine(cluster.superuser_url)
    try:
        async with pooled_engine.begin() as conn:
            await assert_known_not_null_columns(conn, _KNOWN_NOT_NULL_COLUMNS)

            await conn.execute(
                text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
                {"id": tenant_id, "name": name},
            )

            control_fields: dict[str, object] = {"tenant_id": tenant_id}
            if residency is not None:
                control_fields["residency"] = residency
            if isolation_tier != "pooled":
                control_fields["isolation_tier"] = isolation_tier
            if database_alias is not None:
                control_fields["database_alias"] = database_alias
            columns = ", ".join(control_fields)
            placeholders = ", ".join(f":{c}" for c in control_fields)
            await conn.execute(
                text(f"INSERT INTO control.tenants ({columns}) VALUES ({placeholders})"),
                control_fields,
            )

            if dedicated is None:
                identities, memberships, document_ids = await _insert_tenant_data_rows(
                    conn,
                    tenant_id=tenant_id,
                    name=name,
                    seed_identity_id=seed_identity_id,
                    roles=roles,
                    documents=documents,
                    mirror_tenant_row=False,
                )
    finally:
        await pooled_engine.dispose()

    if dedicated is not None:
        data_engine = create_async_engine(dedicated.superuser_url)
        try:
            async with data_engine.begin() as conn:
                await assert_known_not_null_columns(conn, _KNOWN_NOT_NULL_COLUMNS)
                identities, memberships, document_ids = await _insert_tenant_data_rows(
                    conn,
                    tenant_id=tenant_id,
                    name=name,
                    seed_identity_id=seed_identity_id,
                    roles=roles,
                    documents=documents,
                    mirror_tenant_row=True,
                )
        finally:
            await data_engine.dispose()

    return SeededTenant(
        tenant_id=tenant_id,
        name=name,
        residency=residency,
        isolation_tier=isolation_tier,
        identities=identities,
        memberships=memberships,
        document_ids=document_ids,
        cluster=cluster,
        database_alias=database_alias,
        database=dedicated,
    )
