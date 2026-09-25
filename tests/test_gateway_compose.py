"""Rendered compose configuration: the model gateway (LiteLLM) is a required service (ADR-0009,
Spec 7 / #51), not an optional `--profile gateway` add-on.

Skips cleanly when the Docker CLI/daemon isn't available (e.g. some CI runners); real Postgres
role/database isolation for the gateway is covered separately in
tests/test_gateway_bootstrap_integration.py (embedded Postgres, mirrors docker/postgres/01-init.sh).

Rendering itself (Docker availability check, throwaway `.env`, `docker compose config`) is
shared with the other compose tests via tests/compose_helpers.py (Spec 1 / #18).
"""

from __future__ import annotations

import subprocess

import pytest

from tests.compose_helpers import docker_compose_available, render_compose, temporary_env_file

# Never the application's own secrets: the gateway must not receive these even though they sit
# right next to it in .env.
FORBIDDEN_ENV_KEYS = {
    "APP_DB_PASSWORD",
    "APP_OWNER_DB_PASSWORD",
    "POSTGRES_PASSWORD",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_PUBLIC_KEY",
}

pytestmark = pytest.mark.skipif(
    not docker_compose_available(), reason="docker compose not available"
)


@pytest.fixture
def env_file():
    with temporary_env_file() as path:
        yield path


def test_gateway_service_has_no_profile_gate(env_file):
    """`docker compose up` must start the gateway by default — no `--profile gateway` needed."""
    rendered = render_compose()
    litellm = rendered["services"]["litellm"]
    assert "profiles" not in litellm or litellm["profiles"] in (None, [])


def test_gateway_environment_is_minimal_and_isolated(env_file):
    """Only its own database URL, its master credential, and provider credentials its aliases
    call — never the application's DB credential, the owner credential, or the tracing secret."""
    rendered = render_compose()
    litellm_env = rendered["services"]["litellm"]["environment"]

    assert litellm_env["LITELLM_MASTER_KEY"] == "test-master-key"
    assert "DATABASE_URL" in litellm_env
    assert set(litellm_env) & FORBIDDEN_ENV_KEYS == set()


def test_gateway_database_url_is_its_own_and_distinct_from_the_app_database(env_file):
    rendered = render_compose()
    litellm_db_url = rendered["services"]["litellm"]["environment"]["DATABASE_URL"]
    app_db_url = rendered["services"]["api"]["environment"]["DATABASE_URL"]

    assert litellm_db_url != app_db_url
    # Same Postgres cluster, different database name.
    assert litellm_db_url.rsplit("/", 1)[-1] == "gateway"
    assert app_db_url.rsplit("/", 1)[-1] == "app"


def test_gateway_image_is_pinned_not_a_floating_tag(env_file):
    rendered = render_compose()
    image = rendered["services"]["litellm"]["image"]
    tag = image.rsplit(":", 1)[-1]
    assert tag not in {"latest", "main", "main-stable", "stable"}
    # A pinned release tag names a version, e.g. "v1.80.11-stable.1".
    assert any(char.isdigit() for char in tag)


def test_rendering_fails_when_master_key_is_unset(env_file):
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        render_compose({"LITELLM_MASTER_KEY": None})
    assert "LITELLM_MASTER_KEY" in (excinfo.value.stderr or "")
