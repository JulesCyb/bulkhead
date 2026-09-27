"""Seeding a tenant, pooled or dedicated (issue #96 / spec #90, "A6"; dedicated-tier seeding is
issue #97 / "A6-T2"; seeding through the operator commands and the shared gateway fake is issue
#98 / "A6-T3").

`seed_tenant(via_operator=True)` (the default) provisions the tenant through the same
`app.operator.create.create_tenant` the operator tool and its own tests use, with the shared
gateway admin fake (`tests.support.gateway.fake_gateway_admin_client`) injected as `admin_client`
-- so a seeded tenant gets a real `gateway_credential_alias` and a real secret file on disk under
`GATEWAY_CREDENTIALS_DIR` (pointed at a temporary directory by the `environment` fixture,
`tests.support.cluster`), readable through `app.gateway_credentials.read_gateway_credential`
exactly as a real request would read it. `create_tenant` always creates exactly one admin
identity and membership for a fresh tenant -- so does `seed_tenant(via_operator=True)`,
regardless of whether `"admin"` appears in `roles`; `SeededTenant.identities`/`.memberships`
always carry an `"admin"` entry as a result. Any *other* role in `roles` is added on top of it
directly (the same raw statements `via_operator=False` uses for every role), since no operator
command creates a non-admin membership. Documents are always seeded directly too -- no operator
command touches them.

`via_operator=False` is the original, lighter seeding path from #96/#97: every row (including the
tenant's own `tenants`/`control.tenants` pair) written directly, as the cluster's own superuser,
with no gateway credential and no forced admin membership -- for a test that wants exactly the
roles it asked for and does not care about the gateway. `seed_tenant`'s `name` defaults to a
fresh value each call (`f"Seed Co {uuid4().hex[:8]}"`), not a fixed string like the retired
per-file helpers used: `via_operator=True` resolves an existing tenant by exact name
(`app.operator.lookup.resolve_tenant`, `create_tenant`'s own idempotency key) as its very first
step, so two calls sharing one literal default name across two different tests in the same
session-scoped cluster (`tests.support.cluster.cluster`) would silently reconcile onto the same
row instead of seeding two independent tenants.

A pooled tenant's rows all live in the one pooled database `cluster` (the argument every
`seed_tenant` call takes) already points at. A dedicated tenant's own rows (`tenants`,
`memberships`, `documents`) live in its own, separate database instead -- created on the same
embedded cluster, migrated to head with the real runner, exactly as ADR-0002 describes -- while
its control-plane bookkeeping (`control.tenants`: residency, isolation tier, database alias)
stays in the pooled database, the one place the control plane ever lives. `via_operator=True`'s
dedicated path reuses `app.operator.create.create_tenant`'s own dedicated provisioning
(`app.operator.dedicated_db.ensure_dedicated_database`) rather than a second, seeding-only copy of
it -- passing `cluster.superuser_url` itself as `--dedicated-db-admin-url`: the embedded cluster's
own bootstrap superuser already has `CREATEDB`-equivalent privilege on its own server, so no
second `pgserver` instance is needed just to prove a dedicated tenant's database is provisioned
for real.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

import scripts.migrate as migrate_module
from app.context import MeansKind, RequestContext, Role
from app.gateway_provisioning import GatewayAdminClient
from app.operator.create import create_tenant
from app.operator.dedicated_db import generate_database_alias
from app.operator.suspend import SuspendResult, set_tenant_suspended
from tests.support.cluster import (
    Cluster,
    _with_database,
    create_database,
)
from tests.support.gateway import fake_gateway_admin_client

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
    `seed_tenant` was asked for (plus, for `via_operator=True`, the admin identity/membership
    `create_tenant` always creates), any seeded `documents`, and (for `via_operator=True`) the
    gateway credential alias `create_tenant` provisioned and the fake admin client that minted it.

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
    gateway_credential_alias: str | None = None
    gateway_admin_client: GatewayAdminClient | None = None

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

    async def suspend(self) -> SuspendResult:
        """Suspends this tenant through `app.operator.suspend.set_tenant_suspended` -- the same
        function `scripts/operator.py suspend` calls -- against the pooled cluster's owner
        connection: suspension state (`control.tenants.suspended_at`) always lives in the control
        plane, never in a dedicated tenant's own database. A test observes the effect through the
        real session layer (`app.db.session.tenant_session(self.ctx(role))` refusing), not by
        reading this method's return value."""
        return await self._set_suspended(True)

    async def unsuspend(self) -> SuspendResult:
        """The reverse of `suspend()` -- through `app.operator.suspend.set_tenant_suspended`
        with `suspended=False`, exactly as `scripts/operator.py unsuspend` does."""
        return await self._set_suspended(False)

    async def _set_suspended(self, suspended: bool) -> SuspendResult:
        engine = create_async_engine(self.cluster.owner_url)
        try:
            async with engine.begin() as conn:
                return await set_tenant_suspended(conn, str(self.tenant_id), suspended=suspended)
        finally:
            await engine.dispose()


@asynccontextmanager
async def _connection(url: str) -> AsyncIterator[AsyncConnection]:
    engine: AsyncEngine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            yield conn
    finally:
        await engine.dispose()


async def seed_membership(
    cluster: Cluster,
    *,
    tenant_id: uuid.UUID,
    role: Role,
    issuer: str = "seed",
    subject: str | None = None,
) -> tuple[uuid.UUID, uuid.UUID]:
    """A global identity plus its membership of `role` in `tenant_id`, as the owner role.
    Returns `(identity_id, membership_id)`. The seam a test reaches for when it needs a
    membership `seed_tenant`'s own `roles=` argument does not cover -- a second membership of a
    role already requested, for example. Writes as the cluster's own superuser -- same reason as
    `seed_tenant` above. Pass the tenant's own dedicated `Cluster` (`SeededTenant.database`), not
    the pooled one, to add a membership to a dedicated tenant -- its memberships live there.

    `issuer`/`subject` default to the same `'seed'`/the identity's own generated id every other
    seeded identity in this module carries; pass them explicitly for a test that needs an
    identity resolvable under a *specific* issuer (e.g. a tenant's own configured human-IdP
    issuer, distinct from the agent-identity issuer) -- see
    `tests/test_agent_identity_end_to_end_integration.py`, this module's own worked example."""
    engine = create_async_engine(cluster.superuser_url)
    identity_id = uuid.uuid4()
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO control.identities (id, issuer, subject) "
                    "VALUES (:id, :issuer, :sub)"
                ),
                {"id": identity_id, "issuer": issuer, "sub": subject or str(identity_id)},
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


async def seed_document(
    cluster: Cluster,
    *,
    tenant_id: uuid.UUID,
    identity_id: uuid.UUID,
    title: str,
    content: str = "content",
    embedding: str | None = None,
) -> uuid.UUID:
    """One `documents` row with a specific `title` (and, optionally, a specific `embedding`
    literal -- see `vector_literal`) -- the seam for a test that needs a document whose title or
    search ranking the test itself asserts on, distinct from `seed_tenant(documents=...)`'s own
    generic, numbered titles and deterministic embeddings. Writes as the cluster's own
    superuser, exactly like `seed_membership`; `created_by`/`updated_by` are both `identity_id`,
    matching how a real `tenant_session()` write attributes a document (migration 0010)."""
    engine = create_async_engine(cluster.superuser_url)
    document_id = uuid.uuid4()
    try:
        async with engine.begin() as conn:
            if embedding is not None:
                await conn.execute(
                    text(
                        "INSERT INTO documents "
                        "(id, tenant_id, title, content, embedding, created_by, updated_by) "
                        "VALUES (:id, :tid, :title, :content, CAST(:emb AS vector), :who, :who)"
                    ),
                    {
                        "id": document_id,
                        "tid": tenant_id,
                        "title": title,
                        "content": content,
                        "emb": embedding,
                        "who": identity_id,
                    },
                )
            else:
                await conn.execute(
                    text(
                        "INSERT INTO documents (id, tenant_id, title, content, created_by, "
                        "updated_by) VALUES (:id, :tid, :title, :content, :who, :who)"
                    ),
                    {
                        "id": document_id,
                        "tid": tenant_id,
                        "title": title,
                        "content": content,
                        "who": identity_id,
                    },
                )
    finally:
        await engine.dispose()
    return document_id


async def seed_conversation(
    cluster: Cluster,
    *,
    tenant_id: uuid.UUID,
    identity_id: uuid.UUID,
    conversation_id: str,
    last_activity_at: datetime | None = None,
    with_message: bool = False,
) -> None:
    """One `conversations` row -- backdated to `last_activity_at` (ADR-0006) when given, else
    left to the table's own defaults -- and, when `with_message`, one `messages` row alongside
    it. Writes as the cluster's own superuser, exactly like `seed_membership`; a test seeding a
    conversation directly (rather than through a real chat turn) needs full control over both
    `created_by` and, for a retention test, exactly how old it is."""
    engine = create_async_engine(cluster.superuser_url)
    try:
        async with engine.begin() as conn:
            if last_activity_at is not None:
                await conn.execute(
                    text(
                        "INSERT INTO conversations "
                        "(tenant_id, conversation_id, created_by, created_at, last_activity_at) "
                        "VALUES (:tid, :cid, :iid, :ts, :ts)"
                    ),
                    {
                        "tid": tenant_id,
                        "cid": conversation_id,
                        "iid": identity_id,
                        "ts": last_activity_at,
                    },
                )
            else:
                await conn.execute(
                    text(
                        "INSERT INTO conversations (tenant_id, conversation_id, created_by) "
                        "VALUES (:tid, :cid, :iid)"
                    ),
                    {"tid": tenant_id, "cid": conversation_id, "iid": identity_id},
                )
            if with_message:
                await conn.execute(
                    text(
                        "INSERT INTO messages (tenant_id, conversation_id, sequence, payload, "
                        "created_by) VALUES (:tid, :cid, 1, '{}'::jsonb, :iid)"
                    ),
                    {"tid": tenant_id, "cid": conversation_id, "iid": identity_id},
                )
    finally:
        await engine.dispose()


async def set_tenant_retention_days(cluster: Cluster, tenant_id: uuid.UUID, days: int) -> None:
    """Sets this tenant's own `tenants.settings['retention_days']` (ADR-0006) directly, as the
    cluster's own superuser, bypassing whatever `app.operator.create._validate_retention_days`
    (#84) would otherwise reject -- `tenants`' own self-only RLS policy would otherwise block the
    update with no `app.tenant_id` context in scope, and going straight to the row is also the
    only way to get a value the write-side cap would refuse (e.g. one written under an earlier,
    higher `MAX_RETENTION_DAYS`, or before the cap existed at all) into the database for a test.
    Used both to prove a tenant's own (shorter) retention period is honored over the documented
    default, and to prove the retention job's read-side clamp on an over-the-cap stored value
    (`app.tenant_settings.effective_retention_days`)."""
    engine = create_async_engine(cluster.superuser_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE tenants SET settings = jsonb_set(settings, '{retention_days}', "
                    "to_jsonb(CAST(:days AS integer))) WHERE id = :id"
                ),
                {"days": days, "id": tenant_id},
            )
    finally:
        await engine.dispose()


async def set_tenant_identity_issuer(cluster: Cluster, tenant_id: uuid.UUID, issuer: str) -> None:
    """Sets this tenant's own `control.tenants.identity_issuer` (migration 0003) directly, as the
    cluster's own superuser -- no operator command exposes a way to set it. Needed only by a test
    proving a tenant's own configured human-IdP issuer never leaks into an unrelated code path
    (e.g. agent-identity token verification, #49) -- see
    `tests/test_agent_identity_end_to_end_integration.py`."""
    engine = create_async_engine(cluster.superuser_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO control.tenants (tenant_id, identity_issuer) "
                    "VALUES (:id, :issuer) ON CONFLICT (tenant_id) DO UPDATE "
                    "SET identity_issuer = EXCLUDED.identity_issuer"
                ),
                {"id": tenant_id, "issuer": issuer},
            )
    finally:
        await engine.dispose()


def _require_secrets_dir(env_var: str) -> Path:
    """The temporary directory `env_var` currently points at -- raises instead of falling back
    to that variable's production default (`app/db/engine_registry.py`'s
    `/run/secrets/tenant-db(-migrations)`, `app/config.py`'s `gateway_credentials_dir`'s
    `/run/secrets`) when it is unset. Dedicated-tier seeding and `via_operator=True` seeding (of
    either tier) both require the `environment` fixture (`tests.support.cluster`), which sets all
    three -- this is the fail-closed guard against a test that used the bare `cluster` fixture
    instead and would otherwise silently read or write a real secrets path."""
    value = os.environ.get(env_var)
    if not value:
        raise RuntimeError(
            f"seed_tenant() requires {env_var} to already point at a temporary directory -- use "
            "the `environment` fixture (tests.support.cluster), not the bare `cluster` fixture, "
            "so this never falls back to reading or writing that variable's production secrets "
            "path."
        )
    return Path(value)


async def _admin_membership_id(
    conn: AsyncConnection, *, tenant_id: uuid.UUID, identity_id: uuid.UUID
) -> uuid.UUID:
    """The membership id `create_tenant` created for its admin identity -- not itself part of
    `CreateTenantResult`, so seeding reads it back once, against whichever connection/database
    that membership actually lives in (the pooled one for a pooled tenant, the dedicated database
    for a dedicated one -- see `seed_tenant`'s own docstring)."""
    return (
        await conn.execute(
            text("SELECT id FROM memberships WHERE tenant_id = :tid AND identity_id = :iid"),
            {"tid": tenant_id, "iid": identity_id},
        )
    ).scalar_one()


async def _insert_extra_roles_and_documents(
    conn: AsyncConnection,
    *,
    tenant_id: uuid.UUID,
    admin_identity_id: uuid.UUID,
    admin_membership_id: uuid.UUID,
    requested_roles: list[Role],
    extra_roles: list[Role],
    documents: int,
) -> tuple[dict[Role, uuid.UUID], dict[Role, uuid.UUID], list[uuid.UUID]]:
    """The rows no operator command creates, for the `via_operator=True` path: one
    `control.identities` + `memberships` row per role in `extra_roles` (every role in
    `requested_roles` other than `"admin"`, which `create_tenant` already wrote), and
    `documents` documents with a 1536-dim embedding. `identities`/`memberships` always carry an
    `"admin"` entry, seeded or not -- see `seed_tenant`'s own docstring. Document attribution
    (`app.identity_id`, matching `documents.created_by`/`updated_by`, migration 0010) prefers the
    *first* role actually requested in `requested_roles` (matching `via_operator=False`'s own
    convention below), falling back to the admin identity only when no extra role was requested.
    """
    identities: dict[Role, uuid.UUID] = {"admin": admin_identity_id}
    memberships: dict[Role, uuid.UUID] = {"admin": admin_membership_id}
    document_ids: list[uuid.UUID] = []

    for role in extra_roles:
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
        creator_role = requested_roles[0] if requested_roles else None
        creator_id = (
            identities.get(creator_role, admin_identity_id) if creator_role else (admin_identity_id)
        )
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
    """`via_operator=False`'s own rows -- optionally its `tenants` row (`mirror_tenant_row`: a
    pooled tenant's row is inserted by its caller, in the same transaction, before this runs; a
    dedicated tenant's own database needs its own copy, since `memberships`/`documents` there
    foreign-key to it locally), one `control.identities` + `memberships` row per role in
    `roles`, and `documents` documents with a 1536-dim embedding, attributed to the first seeded
    role (or a dedicated seed identity if `roles` is empty). `app.identity_id` is set explicitly
    before the document inserts, exactly like a real `tenant_session()` write would set it, so
    `documents.created_by`/`updated_by` (migration 0010) resolves the same way."""
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


def _default_seed_name() -> str:
    """A fresh default per call -- see this module's own docstring for why a fixed literal
    default is unsafe once `via_operator=True` (the default) resolves an existing tenant by exact
    name as its very first step."""
    return f"Seed Co {uuid.uuid4().hex[:8]}"


async def _seed_tenant_via_operator(
    cluster: Cluster,
    *,
    name: str,
    residency: str,
    roles: list[Role],
    documents: int,
    isolation_tier: str,
    admin_email: str,
    admin_client: GatewayAdminClient,
) -> SeededTenant:
    if isolation_tier == "dedicated":
        _require_secrets_dir("TENANT_DB_MIGRATIONS_SECRETS_DIR")
        _require_secrets_dir("TENANT_DB_SECRETS_DIR")
    _require_secrets_dir("GATEWAY_CREDENTIALS_DIR")

    extra_roles = [role for role in roles if role != "admin"]
    dedicated_admin_url = cluster.superuser_url if isolation_tier == "dedicated" else None

    identities: dict[Role, uuid.UUID] = {}
    memberships: dict[Role, uuid.UUID] = {}
    document_ids: list[uuid.UUID] = []

    owner_engine = create_async_engine(cluster.owner_url)
    try:
        async with owner_engine.begin() as conn:
            await assert_known_not_null_columns(conn, _KNOWN_NOT_NULL_COLUMNS)
            result = await create_tenant(
                conn,
                tenant_name=name,
                residency=residency,
                admin_email=admin_email,
                isolation_tier=isolation_tier,
                dedicated_db_admin_url=dedicated_admin_url,
                admin_client=admin_client,
            )

            if isolation_tier == "pooled":
                admin_membership_id = await _admin_membership_id(
                    conn, tenant_id=result.tenant_id, identity_id=result.identity_id
                )
                identities, memberships, document_ids = await _insert_extra_roles_and_documents(
                    conn,
                    tenant_id=result.tenant_id,
                    admin_identity_id=result.identity_id,
                    admin_membership_id=admin_membership_id,
                    requested_roles=roles,
                    extra_roles=extra_roles,
                    documents=documents,
                )
    finally:
        await owner_engine.dispose()

    dedicated: Cluster | None = None
    if isolation_tier == "dedicated":
        alias = result.database_alias
        assert alias is not None  # enforced by create_tenant for isolation_tier="dedicated"
        migrations_dir = Path(os.environ["TENANT_DB_MIGRATIONS_SECRETS_DIR"])
        app_dir = Path(os.environ["TENANT_DB_SECRETS_DIR"])
        dedicated = Cluster(
            superuser_url=_with_database(cluster.superuser_url, alias),
            owner_url=(migrations_dir / alias).read_text(encoding="utf-8").strip(),
            app_url=(app_dir / alias).read_text(encoding="utf-8").strip(),
        )
        data_engine = create_async_engine(dedicated.superuser_url)
        try:
            async with data_engine.begin() as conn:
                await assert_known_not_null_columns(conn, _KNOWN_NOT_NULL_COLUMNS)
                admin_membership_id = await _admin_membership_id(
                    conn, tenant_id=result.tenant_id, identity_id=result.identity_id
                )
                identities, memberships, document_ids = await _insert_extra_roles_and_documents(
                    conn,
                    tenant_id=result.tenant_id,
                    admin_identity_id=result.identity_id,
                    admin_membership_id=admin_membership_id,
                    requested_roles=roles,
                    extra_roles=extra_roles,
                    documents=documents,
                )
        finally:
            await data_engine.dispose()

    return SeededTenant(
        tenant_id=result.tenant_id,
        name=name,
        residency=residency,
        isolation_tier=isolation_tier,
        identities=identities,
        memberships=memberships,
        document_ids=document_ids,
        cluster=cluster,
        database_alias=result.database_alias,
        database=dedicated,
        gateway_credential_alias=result.gateway_credential_alias,
        gateway_admin_client=admin_client,
    )


async def _seed_tenant_raw(
    cluster: Cluster,
    *,
    name: str,
    residency: str | None,
    roles: list[Role],
    documents: int,
    isolation_tier: str,
) -> SeededTenant:
    """`via_operator=False`: every row written directly, as the cluster's own superuser -- the
    original #96/#97 seeding path. See `seed_tenant`'s own docstring for when to reach for this
    instead of the default."""
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


async def _provision_dedicated_tenant_database(cluster: Cluster, alias: str) -> Cluster:
    """Creates `alias`'s own database on `cluster`'s server (`tests.support.cluster.
    create_database`), migrates it to head with the real runner -- `scripts.migrate.
    migrate_alias`, the exact code path production uses for a dedicated alias -- and writes both
    its owner-role and app-role tenant-secret files where `scripts/migrate.py` and
    `app/db/engine_registry.py` read them respectively, so a test that later routes through
    `tenant_session(ctx)` or runs `scripts/migrate.py` against this alias needs no seam of its
    own. Used only by `via_operator=False`'s dedicated path -- `via_operator=True`'s own dedicated
    path reuses `app.operator.dedicated_db.ensure_dedicated_database` instead (see `seed_tenant`'s
    module docstring)."""
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
    await asyncio.to_thread(migrate_module.migrate_alias, alias)

    app_secret = app_dir / alias
    app_secret.parent.mkdir(parents=True, exist_ok=True)
    app_secret.write_text(dedicated.app_url)

    return dedicated


async def seed_tenant(
    cluster: Cluster,
    *,
    name: str | None = None,
    residency: str | None = None,
    roles: Iterable[Role] = (),
    documents: int = 0,
    isolation_tier: str = "pooled",
    via_operator: bool = True,
    admin_email: str | None = None,
    admin_client: GatewayAdminClient | None = None,
) -> SeededTenant:
    """Seeds one tenant (ADR-0002): a `tenants` row, its `control.tenants` row (residency,
    isolation tier, and -- for a dedicated tenant -- its database alias), one real
    `control.identities` + `memberships` row per role in `roles`, and `documents` documents with
    a 1536-dim embedding. See this module's own docstring for the full contract, in particular
    for what `via_operator` (default `True`) changes: it is the difference between "this went
    through the operator tool's `create` command, gateway credential included" and "these rows
    were written directly, with no gateway credential and no forced admin membership."

    `name` defaults to a fresh value each call, not a fixed string -- see the module docstring for
    why. `admin_email` (only meaningful for `via_operator=True`) defaults the same way.
    `admin_client` (also only meaningful for `via_operator=True`) defaults to a fresh
    `tests.support.gateway.fake_gateway_admin_client()`; pass one explicitly to inspect what it
    recorded (`.fake.generate_requests`, `.fake.delete_requests`, ...) or to make it fail
    (`fail_generate=True`/`fail_deletes_until=`).

    `isolation_tier="dedicated"` additionally creates the tenant's own database, migrated to head
    with the real runner, and writes its owner-role/app-role secret files where
    `scripts/migrate.py`/`app/db/engine_registry.py` read them -- requires the `environment`
    fixture (which points the relevant secrets directories at temporary ones); see the module
    docstring for how the two `via_operator` values differ in *how* that database gets created.
    Its `control.tenants` bookkeeping row (residency, isolation tier, database alias) still lives
    in the pooled `cluster` -- the control plane is never itself sharded -- but its own
    `tenants`/`memberships`/`documents` rows are written into its own dedicated database instead.
    The returned `SeededTenant.database_alias`/`owner_url`/`app_url` describe that database;
    `SeededTenant.ctx()` is unchanged either way -- a real request never knows or cares which
    database serves it.
    """
    if isolation_tier not in ("pooled", "dedicated"):
        raise ValueError(f"seed_tenant: unknown isolation_tier {isolation_tier!r}")

    effective_name = name or _default_seed_name()
    roles_list = list(roles)

    if via_operator:
        effective_residency = residency or "eu"
        effective_admin_email = admin_email or f"seed-{uuid.uuid4()}@example.test"
        effective_admin_client = admin_client or fake_gateway_admin_client()
        return await _seed_tenant_via_operator(
            cluster,
            name=effective_name,
            residency=effective_residency,
            roles=roles_list,
            documents=documents,
            isolation_tier=isolation_tier,
            admin_email=effective_admin_email,
            admin_client=effective_admin_client,
        )

    return await _seed_tenant_raw(
        cluster,
        name=effective_name,
        residency=residency,
        roles=roles_list,
        documents=documents,
        isolation_tier=isolation_tier,
    )


async def seed_dedicated_control_row(cluster: Cluster, *, alias: str) -> uuid.UUID:
    """One `tenants` + `control.tenants` row marking a tenant dedicated to `alias`, with no real
    second database behind it. Enough for anything that only reads the alias column -- the
    control repository's alias enumeration and the migration runner's alias discovery -- without
    the cost of provisioning a database (`seed_tenant(isolation_tier="dedicated")` does that, and
    always migrates the database it creates, which is exactly what a test of the *unmigrated*
    state must avoid)."""
    tenant_id = uuid.uuid4()
    engine = create_async_engine(cluster.superuser_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
                {"id": tenant_id, "name": f"dedicated-{alias}"},
            )
            await conn.execute(
                text(
                    "INSERT INTO control.tenants (tenant_id, isolation_tier, database_alias) "
                    "VALUES (:tid, 'dedicated', :alias)"
                ),
                {"tid": tenant_id, "alias": alias},
            )
    finally:
        await engine.dispose()
    return tenant_id
