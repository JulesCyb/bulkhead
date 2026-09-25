"""Guard tests: SSE framing, the dev-headers environment guard, and residency/embedding
configuration-property tests (ADR-0008, spec 8 / issue #58)."""

from __future__ import annotations

import httpx
import pytest
from pydantic import ValidationError

from app.api.agents import _sse
from app.config import RESIDENCY_ALLOW_LIST, Settings
from app.main import check_auth_mode

_VALID_KWARGS = {"embedding_provider": "openai", "embedding_model": "text-embedding-3-small"}


def test_sse_framing_preserves_newlines():
    assert _sse("hello") == "data: hello\n\n"
    # A delta containing newlines must become multiple data: lines (the client
    # reassembles them), never a raw line without the data: prefix.
    assert _sse("line one\nline two") == "data: line one\ndata: line two\n\n"
    assert _sse("") == "data: \n\n"


def test_dev_headers_refused_outside_dev():
    with pytest.raises(RuntimeError):
        check_auth_mode(Settings(environment="prod", auth_mode="dev-headers"))
    # No raise: dev-headers in dev/test, jwt anywhere.
    check_auth_mode(Settings(environment="dev", auth_mode="dev-headers"))
    check_auth_mode(Settings(environment="test", auth_mode="dev-headers"))
    check_auth_mode(Settings(environment="prod", auth_mode="jwt"))


# --- Residency allow-list and required embedding configuration (issue #58) ---


def test_settings_requires_embedding_provider():
    """No default embedding provider: unset must fail construction, not reach a default
    endpoint. Verified directly against Settings, no network call."""
    with pytest.raises(ValidationError, match="EMBEDDING_PROVIDER"):
        Settings(embedding_provider=None, embedding_model="text-embedding-3-small")


def test_settings_requires_embedding_model():
    with pytest.raises(ValidationError, match="EMBEDDING_MODEL"):
        Settings(embedding_provider="openai", embedding_model=None)


def test_settings_residency_allow_list_has_at_least_two_residencies():
    # Data-driven, not hard-coded per provider: at least two residencies for tests, each with
    # allowed model/gateway host patterns, an allowed embedding endpoint, and a trace sink host.
    assert len(RESIDENCY_ALLOW_LIST) >= 2
    for route in RESIDENCY_ALLOW_LIST.values():
        assert route.model_host_patterns
        assert route.embedding_endpoint
        assert route.trace_sink_host


def test_settings_rejects_unknown_residency():
    with pytest.raises(ValidationError, match="Unknown residency"):
        Settings(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            residency="mars",
        )


def test_settings_valid_configuration_constructs_cleanly():
    settings = Settings(
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        residency="eu",
    )
    assert settings.residency == "eu"
    assert settings.residency_route == RESIDENCY_ALLOW_LIST["eu"]

    # A second, distinct residency also constructs cleanly and resolves its own route.
    other = Settings(
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        residency="us",
    )
    assert other.residency_route != settings.residency_route


def test_settings_backup_retention_days_has_a_documented_default(monkeypatch):
    """Seam 2 (Spec 9 / #72, ADR-0010): the backup-retention window parses from configuration
    with a documented default -- no database involved. `BACKUP_RETENTION_DAYS` overrides it, same
    convention as every other env-backed field on `Settings`."""
    monkeypatch.delenv("BACKUP_RETENTION_DAYS", raising=False)
    default_settings = Settings(
        embedding_provider="openai", embedding_model="text-embedding-3-small", residency="eu"
    )
    assert default_settings.backup_retention_days == 30

    monkeypatch.setenv("BACKUP_RETENTION_DAYS", "14")
    overridden = Settings(
        embedding_provider="openai", embedding_model="text-embedding-3-small", residency="eu"
    )
    assert overridden.backup_retention_days == 14


def test_erasure_backup_horizon_reflects_the_configured_retention_setting():
    """Seam 2: an erasure record's computed backup-horizon date reflects
    `Settings.backup_retention_days` -- a pure function, tested at the Settings level, no
    database or operator-tool connection involved (`app.operator.erase.compute_backup_horizon`)."""
    from datetime import UTC, datetime

    from app.operator.erase import compute_backup_horizon

    settings = Settings(
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        residency="eu",
        backup_retention_days=45,
    )
    now = datetime(2026, 1, 1, tzinfo=UTC)
    horizon = compute_backup_horizon(settings=settings, now=now)
    assert horizon == datetime(2026, 2, 15, tzinfo=UTC)


# --- Fail-closed configuration: no permissive defaults, secrets never printed (issue #14) ---


def test_settings_requires_environment_and_auth_mode(monkeypatch):
    """No code default for either field: a `.env` copied and left unedited must fail to
    construct rather than silently becoming a permissive local configuration (`dev` /
    `dev-headers`). `_env_file=None` bypasses any real `.env` on disk so this only exercises the
    field defaults themselves."""
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("AUTH_MODE", raising=False)
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            _env_file=None,
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )
    missing = {str(e["loc"][0]) for e in exc_info.value.errors()}
    assert {"environment", "auth_mode"} <= missing

    # Supplying both explicitly still constructs cleanly.
    Settings(
        _env_file=None,
        environment="dev",
        auth_mode="dev-headers",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
    )


def test_settings_secrets_never_printed_in_repr_or_str():
    """Every tenant-secret and connection-string field is a SecretStr: repr()/str() of the
    Settings object must never leak a live value, whether printed directly or via logging."""
    live_values = {
        "database_url": "postgresql+asyncpg://app:s3cr3t-db@db.internal:5432/app",
        "litellm_api_key": "litellm-live-value",
        "langfuse_secret_key": "langfuse-live-value",
    }
    settings = Settings(
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        **live_values,
    )
    dump = f"{settings!r}\n{settings}"
    for field, live_value in live_values.items():
        assert live_value not in dump, f"{field} leaked its live value into repr()/str()"
    assert settings.database_url.get_secret_value() == live_values["database_url"]


def test_settings_has_no_direct_provider_api_key_fields():
    """ADR-0009 / ai-app-starter#7 review finding: a raw provider API key on this object would be
    exactly the escape hatch that lets application code build a client that skips the gateway.
    `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` are real environment variables (`.env`, `docker-compose
    .yml`), but they belong to the `litellm` gateway service alone -- `app.config.Settings`, which
    configures the long-running API process, has no field for either."""
    assert "openai_api_key" not in Settings.model_fields
    assert "anthropic_api_key" not in Settings.model_fields


def test_settings_has_no_database_owner_connection_string_field():
    """The migrations/owner DSN is not a field on the application's configuration object at all
    (issue #14 / ADR-0011), so no code path in the long-running API process can ever construct a
    connection with the database owner's privileges."""
    assert "database_url_migrations" not in Settings.model_fields


def test_migration_settings_independent_of_application_settings(monkeypatch):
    """Alembic's own environment module and the operator tool (app/operator/cli.py) resolve the
    owner DSN from app.migration_settings, a source app.main and app.deps never import — never
    from app.config.Settings, which has no such field at all."""
    import inspect

    from app import deps as deps_module
    from app import main as main_module
    from app.migration_settings import MigrationSettings, get_migration_settings

    assert "database_url_migrations" not in Settings.model_fields
    assert "database_url_migrations" in MigrationSettings.model_fields

    monkeypatch.setenv("DATABASE_URL_MIGRATIONS", "postgresql+asyncpg://app_owner@db/app")
    get_migration_settings.cache_clear()
    settings = get_migration_settings()
    assert settings.database_url_migrations.get_secret_value() == (
        "postgresql+asyncpg://app_owner@db/app"
    )
    get_migration_settings.cache_clear()

    for module in (main_module, deps_module):
        assert "migration_settings" not in inspect.getsource(module)


# --- Fail-closed startup and live readiness guard (issue #15 / ADR-0011) ---


def test_create_app_raises_immediately_for_prod_dev_headers_without_any_lifespan_event():
    """The dev-headers guard runs at application-*construction* time: building the app object
    with production-like settings and header-based auth raises immediately, with no ASGI
    lifespan event ever firing — closing the gap where a test harness (or any other way of
    launching the process) that never drives lifespan could skip the guard entirely."""
    from app.main import create_app

    prod_settings = Settings(environment="prod", auth_mode="dev-headers", **_VALID_KWARGS)
    with pytest.raises(RuntimeError, match="dev-headers"):
        create_app(settings=prod_settings)


async def test_lifespan_also_raises_independent_of_the_construction_time_check(monkeypatch):
    """Construction succeeds with dev-like settings. Driving the ASGI lifespan directly
    afterwards, with `get_settings()` now returning production-like, header-auth settings, must
    still raise — proving the lifespan's own guard call is a second, independent execution of
    `check_auth_mode`, not a value cached from what construction saw."""
    from app import main as main_module

    dev_settings = Settings(environment="dev", auth_mode="dev-headers", **_VALID_KWARGS)
    app_obj = main_module.create_app(settings=dev_settings)

    prod_settings = Settings(environment="prod", auth_mode="dev-headers", **_VALID_KWARGS)
    monkeypatch.setattr(main_module, "get_settings", lambda: prod_settings)

    with pytest.raises(RuntimeError, match="dev-headers"):
        async with app_obj.router.lifespan_context(app_obj):
            pass


async def test_lifespan_raises_when_a_substituted_role_rls_guard_fails(monkeypatch):
    """With auth-mode settings that pass cleanly, a substituted, failing role/RLS guard still
    raises during ASGI startup, before the application could ever accept a request."""
    from app import main as main_module

    good_settings = Settings(environment="dev", auth_mode="dev-headers", **_VALID_KWARGS)
    app_obj = main_module.create_app(settings=good_settings)
    monkeypatch.setattr(main_module, "get_settings", lambda: good_settings)

    async def failing_guard() -> None:
        raise RuntimeError("substituted role/RLS guard failure")

    monkeypatch.setattr(main_module, "run_role_rls_guard", failing_guard)

    with pytest.raises(RuntimeError, match="substituted role/RLS guard failure"):
        async with app_obj.router.lifespan_context(app_obj):
            pass


@pytest.fixture
def ready_client():
    from app.main import app as main_app

    transport = httpx.ASGITransport(app=main_app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_ready_endpoint_returns_generic_503_when_guard_fails(ready_client):
    """A substituted, failing guard makes `/ready` answer 503 with a body that names neither
    the offending role nor table — only the server log (`caplog`) sees the real reason."""
    from app.api.health import get_role_rls_guard
    from app.main import app as main_app

    offending_detail = "role app_super_secret is a superuser on table super_secret_table"

    async def failing_guard() -> None:
        raise RuntimeError(offending_detail)

    main_app.dependency_overrides[get_role_rls_guard] = lambda: failing_guard
    try:
        async with ready_client as client:
            response = await client.get("/ready")
    finally:
        main_app.dependency_overrides.pop(get_role_rls_guard, None)

    assert response.status_code == 503
    assert offending_detail not in response.text
    assert "app_super_secret" not in response.text
    assert "super_secret_table" not in response.text


async def test_ready_endpoint_returns_200_when_guard_passes(ready_client):
    from app.api.health import get_role_rls_guard
    from app.main import app as main_app

    async def passing_guard() -> None:
        return None

    main_app.dependency_overrides[get_role_rls_guard] = lambda: passing_guard
    try:
        async with ready_client as client:
            response = await client.get("/ready")
    finally:
        main_app.dependency_overrides.pop(get_role_rls_guard, None)

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


async def test_ready_endpoint_reruns_the_guard_on_every_call_not_cached(ready_client):
    """Two calls to `/ready` must each invoke the guard again — the readiness check must never
    reuse a cached result from an earlier check (boot-time or a previous request)."""
    from app.api.health import get_role_rls_guard
    from app.main import app as main_app

    call_count = 0

    async def counting_guard() -> None:
        nonlocal call_count
        call_count += 1

    main_app.dependency_overrides[get_role_rls_guard] = lambda: counting_guard
    try:
        async with ready_client as client:
            first = await client.get("/ready")
            second = await client.get("/ready")
    finally:
        main_app.dependency_overrides.pop(get_role_rls_guard, None)

    assert first.status_code == 200
    assert second.status_code == 200
    assert call_count == 2


def test_settings_reads_secret_from_secrets_dir(tmp_path):
    """A production deployment can supply a secret as a file under a secrets directory instead
    of (or in addition to) the process environment, so rotating it is a file replacement."""
    dsn = "postgresql+asyncpg://app:from-file@db.internal:5432/app"
    (tmp_path / "database_url").write_text(dsn)
    settings = Settings(
        _env_file=None,
        _secrets_dir=tmp_path,
        environment="dev",
        auth_mode="dev-headers",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
    )
    assert settings.database_url.get_secret_value() == dsn
