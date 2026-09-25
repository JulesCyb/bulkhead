"""Run Alembic migrations against one database alias, or against every alias the control plane
currently knows about (Spec 10 / #76, ADR-0002, ADR-0011).

ADR-0002 decided that migrations run once per database alias rather than once for the whole
deployment: a dedicated tenant's database must never drift from the pooled schema, and
provisioning a brand-new alias must never re-migrate anyone else's database. This script is the
one place that decision is executed:

- With no argument, it brings the pooled database to head first (that is also where the control
  plane's own `control.database_aliases` view lives, so a fresh deployment can bootstrap before
  the view has anything to say), then reads that view and brings every additional alias it names
  to head too.
- With an alias named explicitly, it migrates only that alias's database -- every other
  instance, including the pooled one, is left untouched.

The pooled alias's connection string is `DATABASE_URL_MIGRATIONS` (`app.migration_settings`,
owner role), exactly as `alembic upgrade head` has always used directly. Every other alias's
owner-role connection string is *not* a `Settings`/`MigrationSettings` field, and it is not the
same secret file `app/db/engine_registry.py` reads either -- that file holds the `app` role's
DSN for the running application, which cannot run a migration (it owns nothing). This script
reads a sibling per-alias secret file, one directory over, holding the owner role's DSN instead
(`TENANT_DB_MIGRATIONS_SECRETS_DIR`, default `/run/secrets/tenant-db-migrations`; overridable
for tests). An alias the control plane names with no matching file here fails loudly -- it is
never silently skipped.

    uv run python scripts/migrate.py               # every alias the control plane knows, to head
    uv run python scripts/migrate.py tenant-blue    # only that one alias, to head
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.db.engine_registry import POOLED_ALIAS
from app.migration_settings import get_migration_settings

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ALEMBIC_INI = _REPO_ROOT / "alembic.ini"

# Owner-role DSNs, one file per dedicated alias -- a sibling of, but never the same directory or
# file as, app/db/engine_registry.py's app-role TENANT_DB_SECRETS_DIR: migrations run as the
# owner role, never as `app`.
_DEFAULT_MIGRATIONS_SECRETS_DIR = "/run/secrets/tenant-db-migrations"


class MissingMigrationSecretError(RuntimeError):
    """A dedicated alias has no owner-role connection string on disk.

    Raised instead of silently skipping the alias: an operator who provisions a new alias in the
    control plane but forgets (or has not yet delivered) its migration secret file must see a
    clear failure, not a deployment that quietly leaves that database on an old schema version.
    """

    def __init__(self, alias: str, path: Path) -> None:
        super().__init__(
            f"No owner-role migration secret file for database alias {alias!r} at {path} -- "
            "refusing to skip it silently. Provision the secret file before migrating (or "
            "before routing any tenant to) this alias."
        )
        self.alias = alias


def _migrations_secrets_dir() -> Path:
    return Path(os.environ.get("TENANT_DB_MIGRATIONS_SECRETS_DIR", _DEFAULT_MIGRATIONS_SECRETS_DIR))


def _owner_dsn_for_alias(alias: str) -> str:
    """The owner-role DSN to migrate `alias` with. The pooled alias always uses
    DATABASE_URL_MIGRATIONS; any other alias reads its own file under
    TENANT_DB_MIGRATIONS_SECRETS_DIR, named exactly `alias`."""
    if alias == POOLED_ALIAS:
        return get_migration_settings().database_url_migrations.get_secret_value()

    path = _migrations_secrets_dir() / alias
    try:
        dsn = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise MissingMigrationSecretError(alias, path) from None
    if not dsn:
        raise MissingMigrationSecretError(alias, path)
    return dsn


async def _dedicated_aliases_from_control_plane() -> set[str]:
    """Every alias currently referenced by a tenant in the pooled database's control plane,
    minus the pooled alias itself (that one is always migrated regardless -- see `migrate_all`).

    Reads through `control.enumerate_database_aliases()` (migration 0016), not the plain
    `control.database_aliases` view (0005): `DATABASE_URL_MIGRATIONS` connects as `app_owner`, a
    real NOBYPASSRLS role, and `control.tenants` carries FORCE ROW LEVEL SECURITY, so a
    security_invoker view over it is only readable cross-tenant by a role that bypasses RLS
    outright. The function is a narrow, migration-runner-only escape hatch instead (see its own
    migration for why). Assumes the pooled database is already at head, since both the function
    and the control schema it reads live in migrations this alias must already have applied.
    """
    dsn = get_migration_settings().database_url_migrations.get_secret_value()
    engine = create_async_engine(dsn)
    try:
        async with engine.connect() as conn:
            rows = (
                (
                    await conn.execute(
                        text("SELECT database_alias FROM control.enumerate_database_aliases()")
                    )
                )
                .scalars()
                .all()
            )
    finally:
        await engine.dispose()
    return set(rows) - {POOLED_ALIAS}


def _upgrade_head(dsn: str) -> None:
    config = Config(str(_ALEMBIC_INI))
    config.set_main_option("script_location", str(_REPO_ROOT / "migrations"))
    config.attributes["migration_database_url"] = dsn
    command.upgrade(config, "head")


def migrate_alias(alias: str) -> None:
    """Bring exactly `alias`'s database to head. Never touches any other alias's database."""
    dsn = _owner_dsn_for_alias(alias)
    print(f"migrate: alias {alias!r} -> head")
    _upgrade_head(dsn)


def migrate_all() -> None:
    """Bring the pooled database to head, then every additional alias the control plane (now
    readable, since the pooled database is at head) currently enumerates."""
    migrate_alias(POOLED_ALIAS)
    for alias in sorted(asyncio.run(_dedicated_aliases_from_control_plane())):
        migrate_alias(alias)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) > 1:
        sys.exit("Usage: uv run python scripts/migrate.py [alias]")
    if len(argv) == 1:
        migrate_alias(argv[0])
    else:
        migrate_all()


if __name__ == "__main__":
    main()
