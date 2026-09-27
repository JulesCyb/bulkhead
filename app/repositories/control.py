"""Control-plane data access -- the only module that issues SQL against the `control` schema
(spec A5 / #113, #114). The operator commands (`app/operator/`) and `app/gateway_provisioning.py`
are compositions over this repository now: `create`, `erase`, `suspend`/`unsuspend`, `list`, and
gateway provisioning/revocation call the functions below rather than carrying any SQL of their
own against `control.*`. `tests/test_control_schema_single_path.py` greps for it (parsing every
`text(...)` call under `app/`/`scripts/` with `ast`, not a text grep) and exempts only this file
and, pre-existing and unrelated to spec A5, `app/repositories/agent_identities.py` (see that
test's own docstring).

Two sides, two session kinds:

- **The `app`-role read side** (ADR-0003 / ADR-0011 / issue #22): the narrow, cross-tenant reads
  the `app` role is granted, on a session from `control_session()` (app/db/session.py) -- never
  `tenant_session(ctx)`, and never any other query against `control.*`. `find_by_issuer_and_subject`
  and `get` below are read-only by construction: they issue a single SELECT each and return None on
  no match rather than raising. `get_tenant_record` (Spec 7 / #52, #104) is the one tenant-scoped
  read: the caller's own row through `control.tenants_view` (RLS-filtered to that one tenant) plus
  the tenant's own settings, on a session from `tenant_record_session(tenant_id)` (pooled database,
  `app.tenant_id` set for that one transaction) -- see `app.tenant_record`. It is the only reader of
  residency and the gateway credential alias (#105): model, embedding, and tracing resolution are
  functions of the record it returns, and read nothing themselves. `enumerate_referenced_aliases`
  is also reachable from this side (the fail-closed guard, `app/db/guard.py`, calls it on an
  app-role `control_session()` -- `app` is granted `EXECUTE` on the underlying function by
  migration 0017) as well as from the owner-role side below (the migration runner,
  `scripts/migrate.py`), hence its `AsyncConnection | AsyncSession` parameter.
- **The owner-role write side** (spec A5 / #113, #114, ADR-0010, ADR-0011): every function below
  takes an already-open owner-role `AsyncConnection` (the operator CLI's own transaction, or one
  opened for the duration of a single call by a caller with no transaction of its own -- see
  `app.gateway_provisioning`) and writes or reads `control.tenants` directly rather than through a
  `SECURITY DEFINER` escape hatch, except `set_suspended`, which already has one
  (`control.set_tenant_suspended()`, migration 0024). `control.tenants` carries `FORCE ROW LEVEL
  SECURITY` (migration 0002), which binds the owner role exactly as it binds `app` -- every one of
  those direct reads/writes must first set `app.tenant_id` to the tenant it is about to touch,
  even to read or write that tenant's own row. `_set_owner_tenant_context` below is the one
  private helper that idiom lives in for the pooled database (#114); the pooled-path membership
  write (`app.repositories.memberships.ensure_membership`) relies on the context it leaves on the
  caller's transaction, and the only other owner-role `set_config` is
  `app.operator.dedicated_db`'s, against a dedicated database this repository never reaches. Not
  every owner-role write needs it: `control.identities` (`IdentityRepository.upsert`) and the two
  append-only audit tables (`record_operator_action`, `record_erasure`) carry no `tenant_id`/RLS
  at all (0003, 0004), so those three write with no forced-RLS context.
- **`enumerate_tenants`** (Spec 9 / #68, migration 0012/0024): every tenant's lifecycle facts, for
  the `list` command and the tenant-lookup helper (`app/operator/lookup.py`) alike -- lookup fetches
  the same full enumeration and filters it in Python rather than adding a second, targeted query,
  since `control.enumerate_tenants()` is the one `SECURITY DEFINER` function granted for this
  cross-tenant read and this repository is the only caller of it.

The application never creates, changes, or removes an identity; only the owner-role seed/admin
path does that.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from app.repositories.tenant_settings import TenantSettingsRepository
from app.tenant_record import TenantRecord
from app.tenant_settings import TenantSettings


class Identity(BaseModel):
    """Exactly what `control.identity_lookup` exposes -- never `display_name`/`email`, even
    though `control.identities` carries those columns."""

    id: UUID
    issuer: str
    subject: str


class TenantAuthSettings(BaseModel):
    issuer: str | None
    suspended: bool


class IdentityRepository:
    async def find_by_issuer_and_subject(
        self, session: AsyncSession, *, issuer: str, subject: str
    ) -> Identity | None:
        row = (
            (
                await session.execute(
                    text(
                        "SELECT id, issuer, subject FROM control.identity_lookup "
                        "WHERE issuer = :issuer AND subject = :subject"
                    ),
                    {"issuer": issuer, "subject": subject},
                )
            )
            .mappings()
            .one_or_none()
        )
        return Identity.model_validate(dict(row)) if row is not None else None

    async def get_by_id(self, session: AsyncSession, *, identity_id: UUID) -> Identity | None:
        """The counterpart lookup by id (Spec 6 / #47): needed to mint a token for an agent
        identity, whose (issuer, subject) pair must be read back before it can be embedded as the
        token's own claims -- `find_by_issuer_and_subject` above is for verifying a token
        already presented; this is for building one."""
        row = (
            (
                await session.execute(
                    text("SELECT id, issuer, subject FROM control.identity_lookup WHERE id = :id"),
                    {"id": str(identity_id)},
                )
            )
            .mappings()
            .one_or_none()
        )
        return Identity.model_validate(dict(row)) if row is not None else None

    async def upsert(
        self,
        conn: AsyncConnection,
        *,
        id: UUID,
        issuer: str,
        subject: str,
        email: str,
    ) -> UUID:
        """The owner-role write side (#114, ADR-0003), unlike the two read methods above: insert
        `control.identities`' admin identity for a tenant `app.operator.create.create_tenant`
        provisions, or update-in-place (`ON CONFLICT (issuer, subject)`) if that issuer/subject
        pair already names one -- the same idempotency key `create_tenant`'s own re-run relies on.
        Also used, with an already-known `id`, to mirror that same identity row into a dedicated
        tenant's own database (`app.operator.dedicated_db.ensure_dedicated_membership`):
        its `memberships.identity_id` foreign key needs a local copy even though identity
        resolution at request time always reads the pooled database's copy (this repository's
        app-role side, above). `control.identities` carries no `tenant_id`/RLS (migration 0003),
        so this needs no forced-RLS context, unlike every other owner-role write in this module.
        """
        return (
            await conn.execute(
                text(
                    "INSERT INTO control.identities (id, issuer, subject, display_name, email) "
                    "VALUES (:id, :issuer, :subject, :email, :email) "
                    "ON CONFLICT (issuer, subject) DO UPDATE SET issuer = EXCLUDED.issuer "
                    "RETURNING id"
                ),
                {"id": id, "issuer": issuer, "subject": subject, "email": email},
            )
        ).scalar_one()


class TenantAuthSettingsRepository:
    """Reads a named tenant's configured issuer and suspension state through
    control.tenant_auth_settings() -- the only narrow read permitted for a *specific* tenant
    over a control-plane session that otherwise sets no tenant context at all."""

    async def get(
        self, session: AsyncSession, *, tenant_id: UUID, default_issuer: str | None = None
    ) -> TenantAuthSettings | None:
        row = (
            (
                await session.execute(
                    text(
                        "SELECT identity_issuer, suspended_at "
                        "FROM control.tenant_auth_settings(:tenant_id)"
                    ),
                    {"tenant_id": str(tenant_id)},
                )
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            return None
        return TenantAuthSettings(
            issuer=row["identity_issuer"] or default_issuer,
            suspended=row["suspended_at"] is not None,
        )


_TENANT_RECORD_QUERY = (
    "SELECT isolation_tier, database_alias, residency, suspended_at, gateway_credential_alias "
    "FROM control.tenants_view WHERE tenant_id = :tid"
)


def _record_from_row(
    tenant_id: UUID, row: Mapping[str, object] | None, settings: TenantSettings
) -> TenantRecord:
    """Shared by `get_tenant_record` (app-role) and `get_record` (owner-role) below -- same
    columns, same mapping to a `TenantRecord`, read on two different session kinds."""
    if row is None:
        return TenantRecord(tenant_id=tenant_id, settings=settings)
    pooled = row["isolation_tier"] == "pooled"
    return TenantRecord(
        tenant_id=tenant_id,
        isolation_tier="pooled" if pooled else "dedicated",
        database_alias=None if pooled else row["database_alias"],
        residency=row["residency"],
        suspended_at=row["suspended_at"],
        gateway_credential_alias=row["gateway_credential_alias"],
        settings=settings,
    )


async def _set_owner_tenant_context(conn: AsyncConnection, tenant_id: UUID) -> None:
    """The forced-RLS workaround (spec A5 / #113): `control.tenants` carries `FORCE ROW LEVEL
    SECURITY` (migration 0002), which binds the owner role exactly as it binds `app` -- an
    owner-role connection must set `app.tenant_id` to the tenant it is about to read or write
    before every direct call against `control.tenants`, even to touch that tenant's own row. The
    one place the operator package and `app/gateway_provisioning.py` get that context from on the
    pooled database (#114): they call this repository's own functions instead of setting it
    themselves. The setting lasts for the caller's whole transaction, so the pooled path's
    membership write (`app.repositories.memberships.ensure_membership`, called by
    `app.operator.create` after `create_tenant_record`/`get_record`) relies on it too -- that
    table is forced-RLS as well (code review 2026-09-26). The one owner-role `set_config` outside
    this helper is `app.operator.dedicated_db.ensure_dedicated_membership`'s, against a
    tenant's *dedicated* database, which this repository never reaches. Never used by
    `set_suspended` below, which goes through `control.set_tenant_suspended()` instead -- a
    `SECURITY DEFINER` function that manages its own escape-hatch flag internally and needs no
    `app.tenant_id` at all."""
    await conn.execute(
        text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": str(tenant_id)}
    )


@dataclass(frozen=True, slots=True)
class RoutingState:
    """Exactly the three facts `tenant_session()`'s routing read (`_resolve_tenant_alias`,
    ADR-0002, Spec 10 / #75, Spec 9 / #69) needs to pick which database serves a tenant with no
    already-resolved `TenantRecord` -- deliberately narrower than `TenantRecord`/
    `get_tenant_record`: no settings read, so a caller that only needs to route (a job, a test,
    the stdio MCP fallback) pays for exactly the one query it always did, never a second one
    against `tenants.settings` it does not need. `isolation_tier`/`database_alias`/
    `suspended_at` are all `None` when the control plane has no row for the tenant at all --
    ADR-0002's pooled default."""

    isolation_tier: str | None
    database_alias: str | None
    suspended_at: datetime | None


@dataclass(frozen=True, slots=True)
class TenantSummary:
    """One tenant's lifecycle facts, exactly as `control.enumerate_tenants()` (migration
    0012/0024) reports them -- backs both the `list` command (`app/operator/listing.py`) and the
    tenant-lookup helper (`app/operator/lookup.py`, #114), which filters this same enumeration by
    id or name in Python rather than adding a second, targeted query against a function granted
    for exactly this one cross-tenant read."""

    tenant_id: UUID
    name: str
    isolation_tier: str
    residency: str | None
    database_alias: str | None
    suspended: bool
    suspended_at: datetime | None


@dataclass(frozen=True, slots=True)
class SuspensionOutcome:
    """What `set_suspended` reports: whether the state actually changed (re-suspending an
    already-suspended tenant, or unsuspending an already-active one, is a no-op) and the
    resulting `suspended_at` (`None` once unsuspended)."""

    tenant_id: UUID
    suspended: bool
    changed: bool
    suspended_at: datetime | None


class ControlRepository:
    async def get_tenant_record(self, session: AsyncSession, *, tenant_id: UUID) -> TenantRecord:
        """`tenant_id`'s control-plane record and its validated tenant-editable settings, read in
        the caller's one transaction (#104): one statement against `control.tenants_view`, one
        against `tenants.settings`. The session must have `app.tenant_id` set to `tenant_id`
        (`app.db.session.tenant_record_session`), so RLS limits both reads to that one tenant.

        No control-plane row is ADR-0002's pooled default (with whatever settings the tenant
        has); a pooled row's `database_alias` is reported as `None` whatever the column holds, so
        no consumer can route a pooled tenant by an alias."""
        row = (
            (await session.execute(text(_TENANT_RECORD_QUERY), {"tid": str(tenant_id)}))
            .mappings()
            .one_or_none()
        )
        settings = await TenantSettingsRepository().get_for_tenant(session, tenant_id=tenant_id)
        return _record_from_row(tenant_id, row, settings)

    async def get_routing_state(self, session: AsyncSession, *, tenant_id: UUID) -> RoutingState:
        """The app-role routing read `tenant_session()` falls back to when its context carries no
        already-resolved `TenantRecord` (#104) -- see `RoutingState`'s own docstring. The session
        must already have `app.tenant_id` set to `tenant_id` (that `set_config` call stays in
        `app/db/session.py`'s `_resolve_tenant_alias`: it is the session's own per-request tenant
        scoping, not the owner-role forced-RLS workaround `_set_owner_tenant_context` exists
        for)."""
        row = (
            (
                await session.execute(
                    text(
                        "SELECT isolation_tier, database_alias, suspended_at "
                        "FROM control.tenants_view WHERE tenant_id = :tid"
                    ),
                    {"tid": str(tenant_id)},
                )
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            return RoutingState(isolation_tier=None, database_alias=None, suspended_at=None)
        return RoutingState(
            isolation_tier=row["isolation_tier"],
            database_alias=row["database_alias"],
            suspended_at=row["suspended_at"],
        )

    async def enumerate_referenced_aliases(self, conn: AsyncConnection | AsyncSession) -> list[str]:
        """Every database alias the control plane currently references (Spec 10 / #77): the
        pooled default plus every dedicated alias at least one tenant is assigned to. Backs the
        fail-closed runtime guard's extension to every open engine (`app/db/guard.py`) -- the
        guard always checks the pooled alias itself, and adds whatever this reports on top of it,
        so an empty control plane (zero tenants) still guards the one engine every deployment
        actually opens. Also backs the migration runner's alias enumeration
        (`scripts/migrate.py`), on a raw owner-role connection instead of a session -- hence the
        `AsyncConnection | AsyncSession` parameter: both expose the `execute()` this needs and
        nothing here keys off which one it got.

        Calls `control.enumerate_database_aliases()` (migrations 0016/0017), a `SECURITY DEFINER`
        function that returns alias strings only -- `app` never gains a cross-tenant view of
        `control.tenants` itself. Enumerating aliases is a startup/readiness/migration concern,
        never something a tenant's own request needs, and needs no `app.tenant_id` at all: the
        function manages its own escape-hatch flag internally, exactly like
        `set_tenant_suspended`."""
        rows = await conn.execute(
            text("SELECT database_alias FROM control.enumerate_database_aliases()")
        )
        return [row[0] for row in rows]

    async def create_tenant_record(
        self,
        conn: AsyncConnection,
        tenant_id: UUID,
        *,
        name: str,
        residency: str,
        isolation_tier: str,
        database_alias: str | None,
        settings_json: str = "{}",
    ) -> None:
        """Writes a brand-new tenant's `tenants` row and its `control.tenants` row, atomically on
        the caller's own owner-role transaction -- the same two `INSERT`s
        `app.operator.create.create_tenant` performs for a fresh tenant (#70/#71), now behind
        this repository's one forced-RLS helper instead of that function's own inline
        `set_config` call (#114)."""
        await _set_owner_tenant_context(conn, tenant_id)
        await conn.execute(
            text(
                "INSERT INTO tenants (id, name, settings) "
                "VALUES (:id, :name, CAST(:settings AS jsonb))"
            ),
            {"id": tenant_id, "name": name, "settings": settings_json},
        )
        await conn.execute(
            text(
                "INSERT INTO control.tenants (tenant_id, residency, isolation_tier, "
                "database_alias) VALUES (:tid, :residency, :tier, :alias)"
            ),
            {
                "tid": tenant_id,
                "residency": residency,
                "tier": isolation_tier,
                "alias": database_alias,
            },
        )

    async def set_suspended(
        self, conn: AsyncConnection, tenant_id: UUID, suspended: bool
    ) -> SuspensionOutcome:
        """Flips `tenant_id`'s suspension state through `control.set_tenant_suspended()`
        (migration 0024) -- the one `SECURITY DEFINER` write path the owner role is granted onto
        `control.tenants`. That function manages its own escape-hatch flag internally, so this is
        the one owner-role method here that never calls `_set_owner_tenant_context`. Idempotent:
        re-suspending an already-suspended tenant, or unsuspending an already-active one, reports
        `changed=False`, never an error."""
        row = (
            await conn.execute(
                text(
                    "SELECT changed, suspended_at "
                    "FROM control.set_tenant_suspended(:tid, :suspended)"
                ),
                {"tid": str(tenant_id), "suspended": suspended},
            )
        ).one()
        return SuspensionOutcome(
            tenant_id=tenant_id,
            suspended=suspended,
            changed=row.changed,
            suspended_at=row.suspended_at,
        )

    async def read_gateway_credential_alias(
        self, conn: AsyncConnection, tenant_id: UUID
    ) -> str | None:
        """The gateway-credential alias currently recorded for `tenant_id`, or `None` if no
        `control.tenants` row exists for it yet (never provisioned) or its alias column is unset
        (provisioned but revoked) -- the owner-role counterpart of
        `app.gateway_provisioning._read_alias_from_control_plane`, on the caller's own already-open
        connection instead of a fresh engine per call.

        Deliberately not named `get_gateway_credential_alias`: that name was `ControlRepository`'s
        own app-role method before #105 retired it in favour of reading the alias off the already-
        resolved `TenantRecord` -- `tests/test_residency.py`'s
        `test_residency_and_settings_are_read_from_the_record_only` guards against that exact name
        reappearing anywhere in `app/`, and this is a different method (owner-role, for the
        operator commands' own writes) that only happens to serve a similar fact."""
        await _set_owner_tenant_context(conn, tenant_id)
        row = (
            await conn.execute(
                text("SELECT gateway_credential_alias FROM control.tenants WHERE tenant_id = :tid"),
                {"tid": tenant_id},
            )
        ).first()
        return row[0] if row and row[0] else None

    async def write_gateway_credential_alias(
        self, conn: AsyncConnection, tenant_id: UUID, alias: str | None
    ) -> None:
        """Writes (or clears, `alias=None`) `tenant_id`'s own
        `control.tenants.gateway_credential_alias` -- the owner-role counterpart of
        `app.gateway_provisioning._record_alias_in_control_plane`. `ON CONFLICT` covers both a
        tenant provisioned for the first time (no `control.tenants` row yet) and
        re-provisioning/rotation of one that already has a row. (Named to match
        `read_gateway_credential_alias` above, not `set_gateway_credential_alias`, for the same
        reason that one avoids `get_gateway_credential_alias`.)"""
        await _set_owner_tenant_context(conn, tenant_id)
        await conn.execute(
            text(
                "INSERT INTO control.tenants (tenant_id, gateway_credential_alias) "
                "VALUES (:tid, :alias) "
                "ON CONFLICT (tenant_id) DO UPDATE "
                "SET gateway_credential_alias = EXCLUDED.gateway_credential_alias"
            ),
            {"tid": tenant_id, "alias": alias},
        )

    async def get_record(self, conn: AsyncConnection, tenant_id: UUID) -> TenantRecord:
        """The owner-role counterpart of `get_tenant_record` (#104): the same full `TenantRecord`
        -- control-plane facts plus tenant-editable settings -- read on the caller's own
        already-open owner-role connection (the operator CLI's transaction) instead of a fresh
        `tenant_record_session(tenant_id)`. `create`'s existing-tenant reconciliation and `erase`'s
        own suspension/tier read (#114) are both exactly this shape."""
        await _set_owner_tenant_context(conn, tenant_id)
        row = (
            (await conn.execute(text(_TENANT_RECORD_QUERY), {"tid": str(tenant_id)}))
            .mappings()
            .one_or_none()
        )
        settings = await TenantSettingsRepository().get_for_tenant(conn, tenant_id=tenant_id)
        return _record_from_row(tenant_id, row, settings)

    async def enumerate_tenants(self, conn: AsyncConnection) -> list[TenantSummary]:
        """Every tenant's lifecycle facts (#114), for the `list` command
        (`app/operator/listing.py`) and the tenant-lookup helper (`app/operator/lookup.py`) alike
        -- see `TenantSummary`'s own docstring for why lookup reuses this rather than a second,
        targeted query. Calls `control.enumerate_tenants()` (migration 0012/0024), the same
        narrow, `current_user`-gated, `SECURITY DEFINER` cross-tenant read the retired per-module
        SQL issued directly; needs no `app.tenant_id` at all, exactly like
        `enumerate_referenced_aliases` -- the function manages its own escape-hatch flag
        internally."""
        rows = (
            await conn.execute(text("SELECT * FROM control.enumerate_tenants() ORDER BY name"))
        ).all()
        return [
            TenantSummary(
                tenant_id=row.tenant_id,
                name=row.name,
                isolation_tier=row.isolation_tier,
                residency=row.residency,
                database_alias=row.database_alias,
                suspended=row.suspended,
                suspended_at=row.suspended_at,
            )
            for row in rows
        ]

    async def record_operator_action(
        self, conn: AsyncConnection, *, tenant_id: UUID, action: str, details_json: str
    ) -> None:
        """Writes one row to `control.operator_actions` (migration 0004, #114) -- the owner-role
        counterpart of `app.operator.audit.record_action`, which builds `details_json` (the
        redacted-argument, timing, and outcome payload) and calls this. No `app.tenant_id` needed:
        this table carries no RLS at all, only the append-only-by-grant restriction migration
        0004 puts on `INSERT` (see CLAUDE.md's "Do not touch" list) -- `tenant_id` here is a plain
        column, not a value RLS filters by."""
        await conn.execute(
            text(
                "INSERT INTO control.operator_actions (tenant_id, action, details) "
                "VALUES (:tenant_id, :action, CAST(:details AS jsonb))"
            ),
            {"tenant_id": str(tenant_id), "action": action, "details": details_json},
        )

    async def record_erasure(
        self, conn: AsyncConnection, *, tenant_id: UUID, details_json: str
    ) -> None:
        """Writes one row to `control.tenant_erasures` (migration 0004, #114) -- the owner-role
        counterpart of `app.operator.erase.record_erasure`, which builds `details_json`
        (`EraseResult.as_details()`) and calls this. Deliberately takes `tenant_id` as a plain
        value, not a foreign key, and needs no `app.tenant_id`: this row must document and outlive
        the tenant row `erase_tenant` may have just deleted, exactly as migration 0004 requires,
        and carries no RLS at all -- only the same append-only-by-grant restriction as
        `record_operator_action` above."""
        await conn.execute(
            text(
                "INSERT INTO control.tenant_erasures (tenant_id, details) "
                "VALUES (:tenant_id, CAST(:details AS jsonb))"
            ),
            {"tenant_id": str(tenant_id), "details": details_json},
        )
