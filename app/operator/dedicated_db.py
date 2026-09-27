"""Provision a dedicated tenant's own physical database, idempotently (#71, ADR-0002, ADR-0010,
ADR-0011, #114).

`app.operator.create.create_tenant` calls the two functions here when `isolation_tier="dedicated"`.
They are kept separate from `create_tenant`'s own single owner-role transaction (a `CREATE
DATABASE` cannot run inside a transaction block at all, and provisioning a fresh Postgres
instance is disk/network I/O, not a database write) but follow the same idempotency discipline:
re-running `create` against an already-provisioned dedicated tenant must not attempt to recreate
its database or reapply its migrations, and must not even require `--dedicated-db-admin-url` to
be supplied again.

`ensure_dedicated_database` does everything a managed-Postgres-with-no-init-hook bootstrap needs,
reusing the exact code paths the standalone tools for each step already use rather than
duplicating their logic:

1. `CREATE DATABASE` on the target server, named exactly by the tenant's alias, unless a database
   of that name already exists there.
2. `scripts.provision_roles.provision()` -- the same idempotent `app_owner`/`app` role and grant
   bootstrap `docker/postgres/01-init.sh` performs locally and the standalone script performs
   against a managed provider (Spec 9 / #67) -- run against the freshly created database.
3. Writes the owner-role DSN to this alias's migration-secret file
   (`TENANT_DB_MIGRATIONS_SECRETS_DIR`, the file `scripts/migrate.py` reads, Spec 10 / #76), then
   calls `scripts.migrate.migrate_alias(alias)` -- the exact same call a standalone
   `uv run python scripts/migrate.py <alias>` makes -- to bring it to the current migration head.
4. Writes the `app`-role DSN to this alias's tenant-secret file (`TENANT_DB_SECRETS_DIR`, the
   file `app/db/engine_registry.py` reads at request time, ADR-0002/ADR-0011).

Idempotency is keyed on step 3's migration-secret file: if it already exists, every step above is
assumed already done, and this function returns immediately without opening `admin_url` (which
may not even have been supplied on a re-run).

`ensure_dedicated_membership` writes the tenant's own bookkeeping stub row and a membership of a
given role *into that dedicated database* -- unlike a pooled tenant, whose membership lives in
the same database as the control plane, a dedicated tenant's membership can only ever live in its
own database (`tests/test_tenant_session_routing_integration.py` proves a dedicated tenant's data
is physically absent from the pooled database). It also mirrors the identity row into the
dedicated database's own (otherwise unused) `control.identities` table, through
`app.repositories.control.IdentityRepository.upsert` (#114) rather than SQL of its own:
`memberships.identity_id` foreign-keys to it in every database migrations create it in, even
though identity resolution at request time always reads the pooled database's copy
(`app/repositories/control.py`) -- the dedicated database's copy exists only to satisfy that
per-database foreign key. The `tenants` stub row stays this module's own raw SQL (it is not the
control schema); the membership is written through
`app.repositories.memberships.ensure_membership` (see `ensure_dedicated_membership`'s own
docstring), refusing (`MembershipRoleConflictError`) rather than silently changing an existing
membership of a different role.

`ensure_dedicated_admin_membership` is a thin `role="admin"` wrapper over
`ensure_dedicated_membership` -- `create_tenant`'s own call site and its worked tests predate the
generalization (#83, `app.operator.add_membership`, the operator command that adds a membership
of any role to an already-provisioned tenant).
"""

from __future__ import annotations

import asyncio
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import URL, make_url

import scripts.migrate as migrate_module
import scripts.provision_roles as provision_roles_module
from app.context import Role
from app.db.lifecycle import owner_engine
from app.repositories.control import IdentityRepository
from app.repositories.memberships import (
    MembershipRoleConflictError,
    ensure_membership,
    get_role_owner,
)

# Mirrors app/db/engine_registry.py's own default exactly -- this is the first code path that
# *writes* to that directory rather than only reading it.
_DEFAULT_TENANT_DB_SECRETS_DIR = "/run/secrets/tenant-db"


class MissingDedicatedAdminUrlError(ValueError):
    """Provisioning a *new* dedicated tenant's database, or dropping an existing one
    (`drop_dedicated_database`), needs `--dedicated-db-admin-url`: an admin connection (CREATEDB
    or DROP DATABASE privilege, respectively) to the Postgres server hosting it. Never raised on a
    re-run against an already-provisioned dedicated tenant, or against one already dropped -- see
    module docstring."""


@dataclass(frozen=True, slots=True)
class DedicatedDatabaseResult:
    alias: str
    owner_dsn: str
    outcome: str  # "provisioned" | "already provisioned"


def generate_database_alias(tenant_id: UUID) -> str:
    """A fresh alias for `tenant_id`'s dedicated database -- also used verbatim as the database's
    own name (quoted where used as a SQL identifier). Mirrors
    `app.gateway_provisioning.generate_gateway_credential_alias`'s shape; short enough to stay
    under Postgres's 63-byte identifier limit."""
    return f"tenant-{tenant_id}-{secrets.token_hex(4)}"


def _app_secret_path(alias: str) -> Path:
    directory = Path(os.environ.get("TENANT_DB_SECRETS_DIR", _DEFAULT_TENANT_DB_SECRETS_DIR))
    return directory / alias


def _migrations_secret_path(alias: str) -> Path:
    return migrate_module._migrations_secrets_dir() / alias


def _as_asyncpg_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url[len("postgresql://") :]
    raise ValueError(
        f"Unsupported --dedicated-db-admin-url scheme: {url!r} "
        "(expected postgresql:// or postgresql+asyncpg://)"
    )


def _render_dsn(
    url: URL,
    *,
    database: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> str:
    """A DSN string for `url` with the given overrides, in the exact unencoded shape this
    codebase's own DSNs everywhere else use (`postgresql+asyncpg://user[:pass]@host[:port]/db
    [?query]`, e.g. `tests/test_migrate_alias_integration.py`, `app/db/engine_registry.py`'s
    tenant-secret files) -- never `URL.render_as_string()`/`str(url)`: both percent-encode
    special characters in query values (a unix-socket path passed as `?host=/tmp/...` becomes
    `?host=%2Ftmp%2F...`), and a literal `%` in a DSN then breaks
    `alembic.config.Config.set_main_option` (`ConfigParser`'s `%`-interpolation) the moment
    `scripts.migrate._upgrade_head` hands it that DSN -- exactly the code path this module calls
    into.
    """
    database = url.database if database is None else database
    username = url.username if username is None else username
    password = url.password if password is None else password

    userinfo = username or ""
    if password:
        userinfo += f":{password}"
    hostport = url.host or ""
    if url.port:
        hostport += f":{url.port}"
    query = "&".join(f"{k}={v}" for k, v in url.query.items())
    suffix = f"?{query}" if query else ""
    return f"{url.drivername}://{userinfo}@{hostport}/{database}{suffix}"


def _write_secret(path: Path, dsn: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dsn)
    try:
        path.chmod(0o600)
    except OSError:
        pass  # best-effort, mirrors app.gateway_provisioning.write_gateway_credential_file


async def ensure_dedicated_database(
    *, alias: str, admin_url: str | None
) -> DedicatedDatabaseResult:
    """Idempotently provision (or reconcile) `alias`'s own physical database. See module
    docstring for the full contract."""
    migrations_path = _migrations_secret_path(alias)
    if migrations_path.exists():
        return DedicatedDatabaseResult(
            alias=alias,
            owner_dsn=migrations_path.read_text(encoding="utf-8").strip(),
            outcome="already provisioned",
        )

    if not admin_url:
        raise MissingDedicatedAdminUrlError(
            "creating a new dedicated tenant needs --dedicated-db-admin-url: an admin connection "
            "(CREATEDB privilege) to the Postgres server that will host its database. Not needed "
            "again once that database has been provisioned."
        )

    admin = make_url(_as_asyncpg_url(admin_url))
    db_name = alias
    quoted_db = '"' + db_name.replace('"', '""') + '"'

    # See `_render_dsn`'s docstring for why every DSN below goes through it rather than
    # `URL.render_as_string()`/`str(url)`.
    admin_dsn = _render_dsn(admin)

    # CREATE DATABASE cannot run inside a transaction block -- AUTOCOMMIT, exactly like
    # docker/postgres/01-init.sh's own separate psql invocation for the gateway database.
    async with owner_engine(admin_dsn, isolation_level="AUTOCOMMIT") as admin_engine:
        async with admin_engine.connect() as conn:
            exists = (
                await conn.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": db_name}
                )
            ).first()
            if exists is None:
                await conn.execute(text(f"CREATE DATABASE {quoted_db}"))

    new_db_admin_dsn = _render_dsn(admin, database=db_name)
    await provision_roles_module.provision(new_db_admin_dsn)

    # Same env vars scripts/provision_roles.py itself reads -- the roles it just created (or
    # confirmed) on the new database carry exactly these passwords.
    app_owner_password = os.environ.get("APP_OWNER_DB_PASSWORD", "app_owner")
    app_password = os.environ.get("APP_DB_PASSWORD", "app")
    owner_dsn = _render_dsn(
        admin, database=db_name, username="app_owner", password=app_owner_password
    )
    app_dsn = _render_dsn(admin, database=db_name, username="app", password=app_password)

    _write_secret(migrations_path, owner_dsn)
    # scripts.migrate.migrate_alias is a blocking call that itself calls asyncio.run()
    # internally (alembic has no async API) -- fatal if called directly from a coroutine already
    # running inside an event loop (this one), so it runs in a worker thread instead.
    await asyncio.to_thread(migrate_module.migrate_alias, alias)

    _write_secret(_app_secret_path(alias), app_dsn)

    return DedicatedDatabaseResult(alias=alias, owner_dsn=owner_dsn, outcome="provisioned")


async def drop_dedicated_database(*, alias: str, admin_url: str | None) -> str:
    """Idempotently drop `alias`'s own physical database and remove both of its tenant-secret
    files (Spec 9 / #72, ADR-0010, ADR-0002): the counterpart of `ensure_dedicated_database`, used
    by `app.operator.erase.erase_tenant` for a dedicated tenant. Returns `"removed"` or
    `"already absent"`.

    Idempotency is keyed on the same two secret files `ensure_dedicated_database` writes, in
    reverse: if neither exists, the database is assumed already dropped (or was never
    provisioned) and this returns immediately without needing `admin_url` -- a re-run after a
    partial failure, or a second `erase` invocation entirely, never re-raises on what a previous
    run already removed.

    Otherwise `admin_url` (an admin connection, DROP DATABASE privilege, to the server hosting the
    database) is required: `app_owner` is created `NOCREATEDB`
    (`docker/postgres/01-init.sh`/`scripts/provision_roles.py`) and is not this database's owner
    (the role that ran `CREATE DATABASE` at `create` time is), so it cannot drop its own database.
    Exactly like `--dedicated-db-admin-url` at `create` time, this is used only for this one call
    and never stored.
    """
    migrations_path = _migrations_secret_path(alias)
    app_path = _app_secret_path(alias)
    if not migrations_path.exists() and not app_path.exists():
        return "already absent"

    if not admin_url:
        raise MissingDedicatedAdminUrlError(
            "dropping an existing dedicated tenant's database needs "
            "--dedicated-db-admin-url: an admin connection (DROP DATABASE privilege) to the "
            "Postgres server hosting it."
        )

    admin = make_url(_as_asyncpg_url(admin_url))
    quoted_db = '"' + alias.replace('"', '""') + '"'
    # See `_render_dsn`'s docstring (above) for why this goes through it rather than
    # `URL.render_as_string()`/`str(url)`.
    admin_dsn = _render_dsn(admin)

    # DROP DATABASE cannot run inside a transaction block either -- AUTOCOMMIT, same as
    # `ensure_dedicated_database`'s own CREATE DATABASE.
    async with owner_engine(admin_dsn, isolation_level="AUTOCOMMIT") as admin_engine:
        async with admin_engine.connect() as conn:
            # DROP DATABASE fails outright while any session remains connected to it --
            # terminate any that are (this process's own prior connections to it, a lingering
            # test fixture, etc.) before attempting the drop.
            await conn.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :n AND pid <> pg_backend_pid()"
                ),
                {"n": alias},
            )
            await conn.execute(text(f"DROP DATABASE IF EXISTS {quoted_db}"))

    migrations_path.unlink(missing_ok=True)
    app_path.unlink(missing_ok=True)
    return "removed"


async def ensure_dedicated_membership(
    *,
    owner_dsn: str,
    tenant_id: UUID,
    tenant_name: str,
    tenant_settings_json: str,
    identity_id: UUID,
    issuer: str,
    subject: str,
    email: str,
    role: Role,
) -> str:
    """Write the tenant's bookkeeping stub row and a membership of `role` into its own dedicated
    database (idempotent). Returns `"created"` or `"already exists"`, matching `create_tenant`'s
    own pooled-path vocabulary -- raises `MembershipRoleConflictError` (issue #83) instead of
    either of those if a membership already exists for `identity_id` with a *different* role.

    The `tenants`/`memberships` rows below are this dedicated database's *own* copies, not the
    control schema. The `tenants` stub stays this module's own raw SQL (#114, spec #95's own
    decision); the membership goes through the membership repository's owner-side
    `ensure_membership`/`get_role_owner` (code review 2026-09-26, CLAUDE.md rule 3, #83), and the
    `control.identities` mirror row through `IdentityRepository.upsert`. The tenant context all
    three need is set here, on this dedicated-database connection (see the comment below)."""
    async with owner_engine(owner_dsn) as engine, engine.begin() as conn:
        # The one tenant-context `set_config` outside the control repository, kept here on
        # purpose: this connection is to the tenant's *dedicated* database, which the control
        # repository's forced-RLS helper never reaches, and the `tenants` stub row below,
        # `get_role_owner`, and `ensure_membership` all need it (forced RLS binds the owner role
        # too).
        await conn.execute(
            text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": str(tenant_id)}
        )
        await conn.execute(
            text(
                "INSERT INTO tenants (id, name, settings) "
                "VALUES (:id, :name, CAST(:settings AS jsonb)) "
                "ON CONFLICT (id) DO NOTHING"
            ),
            {"id": tenant_id, "name": tenant_name, "settings": tenant_settings_json},
        )
        # Mirrors the same identity only to satisfy this database's own
        # memberships->control.identities foreign key -- never read back from here (identity
        # resolution always reads the pooled database's copy, app/repositories/control.py).
        await IdentityRepository().upsert(
            conn, id=identity_id, issuer=issuer, subject=subject, email=email
        )
        existing_role = await get_role_owner(conn, tenant_id=tenant_id, identity_id=identity_id)
        if existing_role is not None and existing_role != role:
            raise MembershipRoleConflictError(tenant_id, identity_id, existing_role, role)
        return await ensure_membership(
            conn, tenant_id=tenant_id, identity_id=identity_id, role=role
        )


async def ensure_dedicated_admin_membership(
    *,
    owner_dsn: str,
    tenant_id: UUID,
    tenant_name: str,
    tenant_settings_json: str,
    identity_id: UUID,
    issuer: str,
    subject: str,
    admin_email: str,
) -> str:
    """Thin `role="admin"` wrapper over `ensure_dedicated_membership` above -- `create_tenant`'s
    own call site and its worked tests predate the generalization (#83); see that function's
    docstring for the full contract."""
    return await ensure_dedicated_membership(
        owner_dsn=owner_dsn,
        tenant_id=tenant_id,
        tenant_name=tenant_name,
        tenant_settings_json=tenant_settings_json,
        identity_id=identity_id,
        issuer=issuer,
        subject=subject,
        email=admin_email,
        role="admin",
    )
