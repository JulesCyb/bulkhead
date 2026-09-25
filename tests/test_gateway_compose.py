"""Rendered compose configuration: the model gateway (LiteLLM) is a required service (ADR-0009,
Spec 7 / #51), not an optional `--profile gateway` add-on.

Skips cleanly when the Docker CLI/daemon isn't available (e.g. some CI runners); real Postgres
role/database isolation for the gateway is covered separately in
tests/test_gateway_bootstrap_integration.py (embedded Postgres, mirrors docker/postgres/01-init.sh).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"

# Never the application's own secrets: the gateway must not receive these even though they sit
# right next to it in .env.
FORBIDDEN_ENV_KEYS = {
    "APP_DB_PASSWORD",
    "APP_OWNER_DB_PASSWORD",
    "POSTGRES_PASSWORD",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_PUBLIC_KEY",
}


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "compose", "version"],
            check=True,
            capture_output=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return False
    return True


pytestmark = pytest.mark.skipif(not _docker_available(), reason="docker compose not available")


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    """A throwaway .env so `env_file: .env` on the migrate/api services (unrelated to this
    ticket) resolves; `--env-file` alone does not satisfy a service's own `env_file:` directive.
    Written into the real project directory only for the duration of the test, restored/removed
    after — never left behind (this worktree's own file, not shared with other worktrees).
    """
    real_env = REPO_ROOT / ".env"
    existed = real_env.exists()
    backup = real_env.read_text() if existed else None
    example = (REPO_ROOT / ".env.example").read_text()
    real_env.write_text(example)
    try:
        yield real_env
    finally:
        if existed:
            real_env.write_text(backup)
        else:
            real_env.unlink(missing_ok=True)


def _render(env_file, extra_env: dict[str, str] | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "LITELLM_MASTER_KEY"}
    env.update(extra_env or {})
    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "config", "--format", "json"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, result.args, result.stdout, result.stderr
        )
    return json.loads(result.stdout)


def test_gateway_service_has_no_profile_gate(env_file):
    """`docker compose up` must start the gateway by default — no `--profile gateway` needed."""
    rendered = _render(env_file, {"LITELLM_MASTER_KEY": "test-master-key"})
    litellm = rendered["services"]["litellm"]
    assert "profiles" not in litellm or litellm["profiles"] in (None, [])


def test_gateway_environment_is_minimal_and_isolated(env_file):
    """Only its own database URL, its master credential, and provider credentials its aliases
    call — never the application's DB credential, the owner credential, or the tracing secret."""
    rendered = _render(env_file, {"LITELLM_MASTER_KEY": "test-master-key"})
    litellm_env = rendered["services"]["litellm"]["environment"]

    assert litellm_env["LITELLM_MASTER_KEY"] == "test-master-key"
    assert "DATABASE_URL" in litellm_env
    assert set(litellm_env) & FORBIDDEN_ENV_KEYS == set()


def test_gateway_database_url_is_its_own_and_distinct_from_the_app_database(env_file):
    rendered = _render(env_file, {"LITELLM_MASTER_KEY": "test-master-key"})
    litellm_db_url = rendered["services"]["litellm"]["environment"]["DATABASE_URL"]
    app_db_url = rendered["services"]["api"]["environment"]["DATABASE_URL"]

    assert litellm_db_url != app_db_url
    # Same Postgres cluster, different database name.
    assert litellm_db_url.rsplit("/", 1)[-1] == "gateway"
    assert app_db_url.rsplit("/", 1)[-1] == "app"


def test_gateway_image_is_pinned_not_a_floating_tag(env_file):
    rendered = _render(env_file, {"LITELLM_MASTER_KEY": "test-master-key"})
    image = rendered["services"]["litellm"]["image"]
    tag = image.rsplit(":", 1)[-1]
    assert tag not in {"latest", "main", "main-stable", "stable"}
    # A pinned release tag names a version, e.g. "v1.80.11-stable.1".
    assert any(char.isdigit() for char in tag)


def test_rendering_fails_when_master_key_is_unset(env_file):
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        _render(env_file, {})
    assert "LITELLM_MASTER_KEY" in (excinfo.value.stderr or "")
