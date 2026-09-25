"""Guard tests: SSE framing, the dev-headers environment guard, and residency/embedding
configuration-property tests (ADR-0008, spec 8 / issue #58)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.agents import _sse
from app.config import RESIDENCY_ALLOW_LIST, Settings
from app.main import check_auth_mode


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
        "openai_api_key": "sk-openai-live-value",
        "anthropic_api_key": "sk-anthropic-live-value",
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


def test_settings_has_no_database_owner_connection_string_field():
    """The migrations/owner DSN is not a field on the application's configuration object at all
    (issue #14 / ADR-0011), so no code path in the long-running API process can ever construct a
    connection with the database owner's privileges."""
    assert "database_url_migrations" not in Settings.model_fields


def test_migration_settings_independent_of_application_settings(monkeypatch):
    """Alembic's own environment module and scripts/seed.py resolve the owner DSN from
    app.migration_settings, a source app.main and app.deps never import — never from
    app.config.Settings, which has no such field at all."""
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
