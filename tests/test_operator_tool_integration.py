"""Embedded-Postgres integration test for the operator tool skeleton (Spec 9 / #68): audited
command dispatch, the tenant-lookup helper, and the read-only tenant listing. Pattern:
`tests/test_rls_integration.py`.

Also covers the `create` command (Spec 9 / #70): provisioning a pooled tenant end to end, its
idempotency, up-front residency/model validation, and the audit log. The gateway is never
reached over a real network here either -- `GatewayAdminClient` is always built on
`httpx.MockTransport`, via the shared fake `tests.support.gateway.fake_gateway_admin_client`
(issue #98 / "A6-T3").

Cluster boot, role bootstrap, and tenant seeding come from `tests.support` (issue #96 / spec #90,
"A6"); this file's own private copies of all three were deleted in favor of it.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

REPO_ROOT = Path(__file__).resolve().parent.parent

pgserver = pytest.importorskip("pgserver")

from tests.support import cluster, environment, seed_tenant  # noqa: E402
from tests.support.gateway import fake_gateway_admin_client  # noqa: E402

# `cluster`/`environment` are imported only so pytest can discover them as fixtures from this
# module's namespace (the same pattern `tests/test_support_seeding_integration.py` uses) --
# referenced only by parameter name in the tests below, never called directly.
_ = (cluster, environment)


async def test_listing_reports_every_seeded_tenant(environment):
    from app.operator.listing import list_tenants

    tenant_a = await seed_tenant(
        environment, name="Acme", residency="eu", isolation_tier="pooled", via_operator=False
    )
    tenant_b = await seed_tenant(
        environment,
        name="Globex",
        residency="us",
        isolation_tier="dedicated",
        via_operator=False,
    )
    await tenant_b.suspend()

    engine = create_async_engine(tenant_a.cluster.owner_url)
    try:
        async with engine.begin() as conn:
            summaries = await list_tenants(conn)
    finally:
        await engine.dispose()

    by_id = {s.tenant_id: s for s in summaries}
    assert tenant_a.tenant_id in by_id and tenant_b.tenant_id in by_id

    acme = by_id[tenant_a.tenant_id]
    assert acme.isolation_tier == "pooled"
    assert acme.residency == "eu"
    assert acme.database_alias is None
    assert acme.suspended is False

    globex = by_id[tenant_b.tenant_id]
    assert globex.isolation_tier == "dedicated"
    assert globex.residency == "us"
    assert globex.database_alias == tenant_b.database_alias
    assert globex.suspended is True
    assert globex.suspended_at is not None


async def test_lookup_resolves_by_id_and_by_unambiguous_name(environment):
    from app.operator.lookup import resolve_tenant

    tenant = await seed_tenant(environment, name="Initech", via_operator=False)

    engine = create_async_engine(tenant.cluster.owner_url)
    try:
        async with engine.begin() as conn:
            by_id = await resolve_tenant(conn, str(tenant.tenant_id))
            by_name = await resolve_tenant(conn, "Initech")
    finally:
        await engine.dispose()

    assert by_id.tenant_id == tenant.tenant_id
    assert by_name.tenant_id == tenant.tenant_id


async def test_lookup_rejects_ambiguous_name(environment):
    from app.operator.lookup import AmbiguousTenantNameError, resolve_tenant

    tenant = await seed_tenant(environment, name="Dup Co", via_operator=False)
    await seed_tenant(environment, name="Dup Co", via_operator=False)

    engine = create_async_engine(tenant.cluster.owner_url)
    try:
        async with engine.begin() as conn:
            with pytest.raises(AmbiguousTenantNameError):
                await resolve_tenant(conn, "Dup Co")
    finally:
        await engine.dispose()


async def test_lookup_raises_not_found_for_unknown_id_and_name(environment):
    from app.operator.lookup import TenantNotFoundError, resolve_tenant

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            with pytest.raises(TenantNotFoundError):
                await resolve_tenant(conn, str(uuid.uuid4()))
            with pytest.raises(TenantNotFoundError):
                await resolve_tenant(conn, "no such tenant")
    finally:
        await engine.dispose()


async def test_app_cannot_widen_its_view_with_the_operator_read_flag(environment):
    """Regression (mirrors `test_app_cannot_widen_its_control_plane_view_with_the_migration_read_
    flag` in `tests/test_rls_integration.py` for 0016's identically-shaped flag): `app` can set
    any custom setting in its own session, and a real `app` session querying
    `control.tenants_view`/`tenants` still has `current_user = 'app'` (views check table
    permissions as their owner but do not reassign `current_user`, unlike a `SECURITY DEFINER`
    function). So `control_tenants_operator_read`/`tenants_operator_read` requiring `current_user
    = 'app_owner'`, not just the flag, must still leave `app` seeing only its own tenant even
    after setting `app.control_operator_read` itself."""
    tenant_a = await seed_tenant(environment, name="Flag A", via_operator=False)
    await seed_tenant(environment, name="Flag B", via_operator=False)

    engine = create_async_engine(environment.app_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT set_config('app.control_operator_read', 'true', true)"))
            await conn.execute(
                text("SELECT set_config('app.tenant_id', :tid, true)"),
                {"tid": str(tenant_a.tenant_id)},
            )
            visible = (
                (await conn.execute(text("SELECT tenant_id FROM control.tenants_view")))
                .scalars()
                .all()
            )
            visible_public = (await conn.execute(text("SELECT id FROM tenants"))).scalars().all()
    finally:
        await engine.dispose()
    assert visible == [tenant_a.tenant_id]
    assert visible_public == [tenant_a.tenant_id]


async def _operator_actions(superuser_url: str, action: str) -> list:
    """Only the superuser can read `control.operator_actions` in this test -- `app_owner`
    itself holds INSERT-only on it by grant (migration 0004), exactly as production requires."""
    engine = create_async_engine(superuser_url)
    try:
        async with engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT tenant_id, action, details, performed_at "
                        "FROM control.operator_actions WHERE action = :action "
                        "ORDER BY performed_at"
                    ),
                    {"action": action},
                )
            ).all()
    finally:
        await engine.dispose()
    # asyncpg (via SQLAlchemy's raw text()) may hand back a jsonb column as a decoded dict or
    # as its raw string encoding depending on the driver's codec setup -- normalize either way.
    return [
        (
            tenant_id,
            action,
            json.loads(details) if isinstance(details, str) else details,
            performed_at,
        )
        for tenant_id, action, details, performed_at in rows
    ]


async def test_cli_list_records_the_invocation_in_the_operator_action_log(environment, capsys):
    from app.operator.cli import run_operator

    tenant = await seed_tenant(environment, name="Umbrella", residency="eu", via_operator=False)

    engine = create_async_engine(environment.owner_url)
    exit_code = await run_operator(["list"], engine=engine)
    await engine.dispose()
    assert exit_code == 0

    # Golden output (spec A5 / #115, acceptance criterion 2): byte-identical to what the retired
    # `_run_list` printed directly, for this test's own tenant -- `environment`'s underlying
    # cluster is shared across this file, so `list`'s real output also names every tenant an
    # earlier test seeded; this test's own line is checked verbatim, not the whole listing.
    assert (
        f"{tenant.tenant_id}  'Umbrella'                      tier=pooled     "
        "residency=eu      alias=-             suspended=False"
    ) in capsys.readouterr().out.splitlines()

    rows = await _operator_actions(environment.superuser_url, "list")
    assert len(rows) >= 1
    tenant_id, action, details, performed_at = rows[-1]
    # `list` targets every tenant, not one -- the documented nil-UUID sentinel, never NULL
    # (control.operator_actions.tenant_id is NOT NULL).
    assert str(tenant_id) == "00000000-0000-0000-0000-000000000000"
    assert action == "list"
    assert details["outcome"].startswith("ok: listed ")
    assert "operator" in details
    assert "started_at" in details and "finished_at" in details
    assert isinstance(details["duration_ms"], int)
    assert performed_at is not None


async def test_cli_records_a_failed_invocation_too(environment):
    """Even a failed command is recorded, with its error, not silently dropped -- driven through
    a real failure (an unknown tenant), not a monkeypatched private dispatch entry (spec A5 /
    #115: no test reaches `_COMMANDS`)."""
    from app.operator.cli import run_operator

    unknown_id = str(uuid.uuid4())
    engine = create_async_engine(environment.owner_url)
    exit_code = await run_operator(["suspend", unknown_id], engine=engine)
    await engine.dispose()
    assert exit_code == 1

    rows = await _operator_actions(environment.superuser_url, "suspend")
    _, _, details, _ = rows[-1]
    assert details["outcome"] == "error"
    assert "TenantNotFoundError" in details["error"]


async def test_suspend_sets_the_flag_and_timestamp_and_reruns_as_a_no_op(environment):
    """Spec 9 / #69, ADR-0010, seam 1: `suspend` sets `suspended`/`suspended_at`, and running it
    again against an already-suspended tenant reports a no-op, not an error."""
    tenant = await seed_tenant(environment, name="Suspend Co", via_operator=False)

    first = await tenant.suspend()
    second = await tenant.suspend()

    assert first.changed is True
    assert first.suspended is True
    assert first.suspended_at is not None

    assert second.changed is False
    assert second.suspended is True
    assert second.suspended_at == first.suspended_at


async def test_unsuspend_restores_the_tenant_with_nothing_reprovisioned(environment):
    """`unsuspend` clears `suspended_at` and is itself a no-op when the tenant is already
    active -- and never touches isolation tier, database alias, or residency."""
    from app.operator.listing import list_tenants

    tenant = await seed_tenant(
        environment,
        name="Restore Co",
        residency="us",
        isolation_tier="dedicated",
        via_operator=False,
    )
    await tenant.suspend()

    result = await tenant.unsuspend()
    no_op = await tenant.unsuspend()

    engine = create_async_engine(tenant.cluster.owner_url)
    try:
        async with engine.begin() as conn:
            summaries = {t.tenant_id: t for t in await list_tenants(conn)}
    finally:
        await engine.dispose()

    assert result.changed is True
    assert result.suspended is False
    assert result.suspended_at is None

    assert no_op.changed is False

    restored = summaries[tenant.tenant_id]
    assert restored.suspended is False
    assert restored.isolation_tier == "dedicated"
    assert restored.database_alias == tenant.database_alias
    assert restored.residency == "us"


async def test_suspend_creates_a_control_plane_row_for_a_pooled_tenant_with_none_yet(
    environment,
):
    """A tenant provisioned only through the pooled default (ADR-0002, `scripts/seed.py`'s
    original path) has no `control.tenants` row at all -- suspending it must not error, it
    creates one. `seed_tenant` always writes a `control.tenants` row, so this one edge case is
    still seeded by hand -- there is no seam of the shared package for "a tenant with no
    control-plane row at all"."""
    from app.operator.suspend import set_tenant_suspended

    tenant_id = uuid.uuid4()
    engine = create_async_engine(environment.superuser_url)
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant_id, "name": "No Control Row Co"},
        )
    await engine.dispose()

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            result = await set_tenant_suspended(conn, str(tenant_id), suspended=True)
    finally:
        await engine.dispose()

    assert result.changed is True
    assert result.suspended is True
    assert result.suspended_at is not None


async def test_app_cannot_widen_its_control_plane_write_with_the_operator_write_flag(
    environment,
):
    """Mirrors `test_app_cannot_widen_its_view_with_the_operator_read_flag`: `app` setting
    `app.control_operator_write` itself must not grant it write access to `control.tenants` --
    `current_user = 'app_owner'` is what actually restricts the escape hatch, not the flag alone."""
    tenant = await seed_tenant(environment, name="Write Flag Co", via_operator=False)

    engine = create_async_engine(environment.app_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text("SELECT set_config('app.control_operator_write', 'true', true)")
            )
            with pytest.raises(Exception):  # noqa: B017 - asyncpg's InsufficientPrivilegeError
                await conn.execute(
                    text("UPDATE control.tenants SET suspended_at = now() WHERE tenant_id = :tid"),
                    {"tid": str(tenant.tenant_id)},
                )
    finally:
        await engine.dispose()


async def test_cli_suspend_is_idempotent_and_records_both_invocations_in_the_audit_log(
    environment, capsys
):
    from app.operator.cli import run_operator

    tenant = await seed_tenant(environment, name="Audited Suspend Co", via_operator=False)

    argv = ["suspend", str(tenant.tenant_id)]
    first_engine = create_async_engine(environment.owner_url)
    assert await run_operator(argv, engine=first_engine) == 0
    await first_engine.dispose()
    first_printed = capsys.readouterr().out

    second_engine = create_async_engine(environment.owner_url)
    assert await run_operator(argv, engine=second_engine) == 0
    await second_engine.dispose()
    second_printed = capsys.readouterr().out

    # Golden output (spec A5 / #115): byte-identical to what the retired
    # `_run_suspend_or_unsuspend` printed directly.
    assert first_printed == f"ok: {tenant.name!r} ({tenant.tenant_id}) is now suspended\n"
    assert second_printed == (
        f"no-op: {tenant.name!r} ({tenant.tenant_id}) was already suspended\n"
    )

    rows = await _operator_actions(environment.superuser_url, "suspend")
    assert len(rows) >= 2
    first_tenant_id, _, first_details, _ = rows[-2]
    second_tenant_id, _, second_details, _ = rows[-1]
    assert str(first_tenant_id) == str(tenant.tenant_id)
    assert str(second_tenant_id) == str(tenant.tenant_id)
    assert first_details["outcome"].startswith("ok: ")
    assert second_details["outcome"].startswith("no-op: ")


async def test_cli_unsuspend_records_the_invocation_by_tenant_name(environment, capsys):
    from app.operator.cli import run_operator

    tenant = await seed_tenant(environment, name="Named Unsuspend Co", via_operator=False)
    await tenant.suspend()

    engine = create_async_engine(environment.owner_url)
    exit_code = await run_operator(["unsuspend", "Named Unsuspend Co"], engine=engine)
    await engine.dispose()
    assert exit_code == 0
    assert (
        capsys.readouterr().out == f"ok: {tenant.name!r} ({tenant.tenant_id}) is now unsuspended\n"
    )

    rows = await _operator_actions(environment.superuser_url, "unsuspend")
    tenant_id_logged, _, details, _ = rows[-1]
    assert str(tenant_id_logged) == str(tenant.tenant_id)
    assert details["outcome"].startswith("ok: ")


def test_cli_has_no_flag_or_env_var_for_an_alternate_connection_string():
    """Acceptance: the tool connects only as the owner role's DSN; nothing accepts or falls back
    to a superuser (or any other) connection string."""
    from app.operator.cli import build_parser

    parser = build_parser()
    option_strings = {
        opt for action in parser._actions for opt in getattr(action, "option_strings", [])
    }
    assert not any("dsn" in opt.lower() or "database" in opt.lower() for opt in option_strings)


_RETIRED_PRIVATE_DISPATCH_PATTERN = re.compile(
    r"from app\.operator\.cli import _run|cli_module\._run|_COMMANDS"
)


def test_no_test_imports_the_operator_clis_private_dispatch_function():
    """Acceptance (spec A5 / #115): `run_operator(argv, *, engine=None, admin_client=None)` is
    the one entry point every operator test drives -- dispatch (`_run`, `_COMMANDS`) stays
    private, so no test file may import or monkeypatch either. `main(argv=None)` is the
    synchronous script wrapper `scripts/operator.py` calls (`asyncio.run(run_operator(argv))`,
    nothing else) -- not itself a seam a test needs. Mirrors
    `grep -rn 'from app.operator.cli import _run\\|cli_module\\._run\\|_COMMANDS' tests/` finding
    nothing, in pure Python rather than depending on the `grep` binary. This file itself (naming
    the retired pattern above, in prose, for exactly this test) is the one file the scan skips --
    every other file in `tests/` is checked, this one included would only ever flag itself."""
    this_file = Path(__file__).resolve()
    for path in (REPO_ROOT / "tests").rglob("*.py"):
        if path.resolve() == this_file:
            continue
        text = path.read_text(encoding="utf-8")
        match = _RETIRED_PRIVATE_DISPATCH_PATTERN.search(text)
        assert match is None, f"{path} still reaches the operator CLI's private dispatch: {match}"


def test_cli_module_never_imports_the_application_settings_object():
    """`app.config.Settings` is the long-running API's settings object; the operator tool must
    only ever read `app.migration_settings` (the owner DSN)."""
    import ast

    import app.operator.cli as cli_module

    with open(cli_module.__file__) as f:
        tree = ast.parse(f.read())

    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert "app.config" not in imported_modules
    assert imported_modules & {"app.migration_settings"}


async def test_create_provisions_control_plane_credential_and_admin_membership(
    environment, tmp_path
):
    """Acceptance (#70): `create` writes the control-plane record (isolation tier, residency,
    database alias), mints and writes a gateway-credential secret file, and creates the first
    admin membership for the named identity -- all in one call."""
    from app.config import Settings
    from app.operator.create import create_tenant

    settings = Settings(gateway_credentials_dir=str(tmp_path))
    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            result = await create_tenant(
                conn,
                tenant_name="Create Co",
                residency="eu",
                admin_email="admin@create.test",
                settings=settings,
                admin_client=fake_gateway_admin_client(key="sk-create-co"),
            )
    finally:
        await engine.dispose()

    assert result.control_plane == "created"
    assert result.admin_membership == "created"
    assert result.gateway_credential == "provisioned"
    assert (tmp_path / result.gateway_credential_alias).read_text() == "sk-create-co"

    verify_engine = create_async_engine(environment.superuser_url)
    try:
        async with verify_engine.connect() as conn:
            tenant_row = (
                (
                    await conn.execute(
                        text(
                            "SELECT residency, isolation_tier, database_alias, "
                            "gateway_credential_alias FROM control.tenants WHERE tenant_id = :tid"
                        ),
                        {"tid": result.tenant_id},
                    )
                )
                .mappings()
                .one()
            )
            membership_row = (
                await conn.execute(
                    text(
                        "SELECT role FROM memberships WHERE tenant_id = :tid AND identity_id = :iid"
                    ),
                    {"tid": result.tenant_id, "iid": result.identity_id},
                )
            ).one()
    finally:
        await verify_engine.dispose()

    assert tenant_row["residency"] == "eu"
    assert tenant_row["isolation_tier"] == "pooled"
    assert tenant_row["database_alias"] is None
    assert tenant_row["gateway_credential_alias"] == result.gateway_credential_alias
    assert membership_row.role == "admin"


async def test_create_is_idempotent_on_rerun(environment, tmp_path):
    """Acceptance (#70): re-running `create` against the same tenant name performs none of the
    three steps again and reports each as already in place. A second, differently-keyed fake
    admin client proves the gateway is never called a second time -- if it were, the freshly
    minted key would overwrite the alias's file and the final assertion would fail."""
    from app.config import Settings
    from app.operator.create import create_tenant

    settings = Settings(gateway_credentials_dir=str(tmp_path))

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            first = await create_tenant(
                conn,
                tenant_name="Idempotent Co",
                residency="us",
                admin_email="admin@idempotent.test",
                settings=settings,
                admin_client=fake_gateway_admin_client(key="sk-idempotent"),
            )
    finally:
        await engine.dispose()

    assert (first.control_plane, first.admin_membership, first.gateway_credential) == (
        "created",
        "created",
        "provisioned",
    )

    second_engine = create_async_engine(environment.owner_url)
    try:
        async with second_engine.begin() as conn:
            second = await create_tenant(
                conn,
                tenant_name="Idempotent Co",
                residency="us",
                admin_email="admin@idempotent.test",
                settings=settings,
                admin_client=fake_gateway_admin_client(key="sk-should-not-be-minted"),
            )
    finally:
        await second_engine.dispose()

    assert second.tenant_id == first.tenant_id
    assert second.identity_id == first.identity_id
    assert (second.control_plane, second.admin_membership, second.gateway_credential) == (
        "already exists",
        "already exists",
        "already provisioned",
    )
    assert second.gateway_credential_alias == first.gateway_credential_alias
    assert (tmp_path / first.gateway_credential_alias).read_text() == "sk-idempotent"


async def test_create_rejects_unrecognized_residency_before_any_write(environment, tmp_path):
    from app.config import Settings
    from app.operator.create import UnrecognizedResidencyError, create_tenant

    settings = Settings(gateway_credentials_dir=str(tmp_path))
    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            with pytest.raises(UnrecognizedResidencyError):
                await create_tenant(
                    conn,
                    tenant_name="Should Not Exist",
                    residency="mars",
                    admin_email="nobody@example.test",
                    settings=settings,
                    admin_client=fake_gateway_admin_client(),
                )
    finally:
        await engine.dispose()

    verify_engine = create_async_engine(environment.superuser_url)
    try:
        async with verify_engine.connect() as conn:
            count = (
                await conn.execute(
                    text("SELECT count(*) FROM tenants WHERE name = 'Should Not Exist'")
                )
            ).scalar_one()
    finally:
        await verify_engine.dispose()
    assert count == 0


async def test_create_rejects_unrecognized_model_before_any_write(environment, tmp_path):
    from app.config import Settings
    from app.operator.create import UnrecognizedModelError, create_tenant

    settings = Settings(gateway_credentials_dir=str(tmp_path))
    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            with pytest.raises(UnrecognizedModelError):
                await create_tenant(
                    conn,
                    tenant_name="Should Not Exist Either",
                    residency="eu",
                    admin_email="nobody2@example.test",
                    model="gpt-nonexistent",
                    settings=settings,
                    admin_client=fake_gateway_admin_client(),
                )
    finally:
        await engine.dispose()

    verify_engine = create_async_engine(environment.superuser_url)
    try:
        async with verify_engine.connect() as conn:
            count = (
                await conn.execute(
                    text("SELECT count(*) FROM tenants WHERE name = 'Should Not Exist Either'")
                )
            ).scalar_one()
    finally:
        await verify_engine.dispose()
    assert count == 0


async def test_create_rejects_retention_days_above_the_maximum_before_any_write(
    environment, tmp_path
):
    """#84 (ADR-0006, GDPR Art. 5(1)(e)): a per-tenant `retention_days` override above this
    deployment's `MAX_RETENTION_DAYS` is refused before the control-plane record, the credential,
    or the membership is touched -- same shape as the model-allow-list rejection above."""
    from app.config import Settings
    from app.operator.create import RetentionDaysExceedsMaximumError, create_tenant

    settings = Settings(gateway_credentials_dir=str(tmp_path), max_retention_days=365)
    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            with pytest.raises(RetentionDaysExceedsMaximumError):
                await create_tenant(
                    conn,
                    tenant_name="Should Not Exist Retention",
                    residency="eu",
                    admin_email="nobody3@example.test",
                    retention_days=10_000,
                    settings=settings,
                    admin_client=fake_gateway_admin_client(),
                )
    finally:
        await engine.dispose()

    verify_engine = create_async_engine(environment.superuser_url)
    try:
        async with verify_engine.connect() as conn:
            count = (
                await conn.execute(
                    text("SELECT count(*) FROM tenants WHERE name = 'Should Not Exist Retention'")
                )
            ).scalar_one()
    finally:
        await verify_engine.dispose()
    assert count == 0


async def test_create_accepts_a_retention_days_override_at_or_below_the_maximum(
    environment, tmp_path
):
    """The write side only rejects *above* the cap -- a value at or below it is written through
    to `tenants.settings['retention_days']` like any other tenant setting."""
    from app.operator.create import create_tenant

    engine = create_async_engine(environment.owner_url)
    try:
        async with engine.begin() as conn:
            result = await create_tenant(
                conn,
                tenant_name="Fine Retention Co",
                residency="eu",
                admin_email="fine@example.test",
                retention_days=30,
                admin_client=fake_gateway_admin_client(),
            )
    finally:
        await engine.dispose()

    verify_engine = create_async_engine(environment.superuser_url)
    try:
        async with verify_engine.connect() as conn:
            stored = (
                await conn.execute(
                    text("SELECT settings FROM tenants WHERE id = :tid"),
                    {"tid": result.tenant_id},
                )
            ).scalar_one()
    finally:
        await verify_engine.dispose()
    assert stored["retention_days"] == 30


async def test_cli_create_rejects_retention_days_above_the_maximum(
    environment, tmp_path, monkeypatch
):
    """The write seam through the real CLI dispatch (`run_operator`, spec A5 / #115):
    `--retention-days` above `MAX_RETENTION_DAYS` fails the command (nonzero exit) and leaves no
    tenant behind, same as an unrecognized `--model` or `--residency` would."""
    from app import config
    from app.operator.cli import run_operator

    monkeypatch.setenv("GATEWAY_CREDENTIALS_DIR", str(tmp_path))
    config.get_settings.cache_clear()

    engine = create_async_engine(environment.owner_url)
    try:
        exit_code = await run_operator(
            [
                "create",
                "CLI Retention Reject Co",
                "--residency",
                "eu",
                "--admin-email",
                "reject@cli.test",
                "--retention-days",
                "9999",
            ],
            engine=engine,
            admin_client=fake_gateway_admin_client(key="sk-cli-reject"),
        )
    finally:
        await engine.dispose()
        config.get_settings.cache_clear()

    assert exit_code != 0

    verify_engine = create_async_engine(environment.superuser_url)
    try:
        async with verify_engine.connect() as conn:
            count = (
                await conn.execute(
                    text("SELECT count(*) FROM tenants WHERE name = 'CLI Retention Reject Co'")
                )
            ).scalar_one()
    finally:
        await verify_engine.dispose()
    assert count == 0


async def test_cli_create_records_the_invocation_in_the_operator_action_log(
    environment, tmp_path, monkeypatch, capsys
):
    """Acceptance (#70): a `create` invocation through the real CLI dispatch is recorded in the
    operator-action log, with secrets redacted from the logged arguments (none of `create`'s own
    arguments are secret-shaped, so this also proves ordinary arguments still show up plainly).
    The gateway admin client is injected straight into `run_operator()` (spec A5 / #115), not
    monkeypatched onto `app.operator.create.build_admin_client`."""
    from app import config
    from app.operator.cli import run_operator

    monkeypatch.setenv("GATEWAY_CREDENTIALS_DIR", str(tmp_path))
    config.get_settings.cache_clear()

    engine = create_async_engine(environment.owner_url)
    exit_code = await run_operator(
        ["create", "CLI Co", "--residency", "eu", "--admin-email", "admin@cli.test"],
        engine=engine,
        admin_client=fake_gateway_admin_client(key="sk-cli"),
    )
    await engine.dispose()
    config.get_settings.cache_clear()
    assert exit_code == 0

    rows = await _operator_actions(environment.superuser_url, "create")
    assert len(rows) >= 1
    tenant_id, action, details, performed_at = rows[-1]
    assert action == "create"
    assert details["outcome"].startswith("ok: tenant ")
    assert details["args"]["tenant_name"] == "CLI Co"
    assert details["args"]["admin_email"] == "admin@cli.test"
    assert performed_at is not None

    verify_engine = create_async_engine(environment.superuser_url)
    try:
        async with verify_engine.connect() as conn:
            membership_row = (
                await conn.execute(
                    text("SELECT identity_id FROM memberships WHERE tenant_id = :tid"),
                    {"tid": tenant_id},
                )
            ).one()
            alias = (
                await conn.execute(
                    text(
                        "SELECT gateway_credential_alias FROM control.tenants "
                        "WHERE tenant_id = :tid"
                    ),
                    {"tid": tenant_id},
                )
            ).scalar_one()
    finally:
        await verify_engine.dispose()

    # Golden output (spec A5 / #115, acceptance criterion 2): byte-identical to what the retired
    # `_run_create` printed directly, reconstructed from independently-queried database state
    # (not from the result object under test).
    identity_id = membership_row.identity_id
    assert capsys.readouterr().out == (
        f"MCP_TENANT_ID={tenant_id}\n"
        f"MCP_IDENTITY_ID={identity_id}\n"
        f"Gateway credential alias: {alias}\n"
        "control-plane record: created; gateway credential: provisioned; "
        "admin membership: created\n"
        "\n"
        f"curl -H 'X-Identity-Id: {identity_id}' "
        f"http://localhost:8000/v1/t/{tenant_id}/agents/assistant/run ...\n"
    )
