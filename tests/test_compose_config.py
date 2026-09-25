"""Rendered `docker-compose.yml` configuration (Spec 9 / #67): the standalone
`scripts/provision_roles.py` role-provisioning script must not be wired into the compose stack's
automatic startup — an operator on managed Postgres runs it by hand, once, outside compose
entirely. Skipped when the Docker CLI/daemon isn't available."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _docker_compose_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "compose", "version"], capture_output=True, check=True, timeout=10
        )
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _docker_compose_available(), reason="docker compose CLI/daemon not available"
)


@pytest.fixture
def rendered_config(tmp_path, monkeypatch):
    """Renders docker-compose.yml with `docker compose config`. `env_file: .env` is a fixed
    reference in the compose file itself (not resolved by --env-file), so a placeholder .env is
    supplied only if the repo doesn't already have a real one — never overwritten."""
    env_path = REPO_ROOT / ".env"
    created = False
    if not env_path.exists():
        shutil.copy(REPO_ROOT / ".env.example", env_path)
        created = True
    try:
        result = subprocess.run(
            ["docker", "compose", "config", "--format", "json"],
            cwd=REPO_ROOT,
            env={**os.environ, "LITELLM_MASTER_KEY": "test-master-key"},
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        if created:
            env_path.unlink(missing_ok=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_no_service_runs_provision_roles(rendered_config):
    for name, service in rendered_config["services"].items():
        command = service.get("command")
        entrypoint = service.get("entrypoint")
        for field in (command, entrypoint):
            if field is None:
                continue
            text = " ".join(field) if isinstance(field, list) else str(field)
            assert "provision_roles" not in text, f"service {name!r} runs provision_roles.py"


def test_known_services_only(rendered_config):
    """Documents the expected automatic-startup surface, so a future service that does invoke
    the script by accident is caught even if it's not literally named "provision_roles". The
    litellm gateway is a required service since #51, so it is part of the default set."""
    assert set(rendered_config["services"]) == {"postgres", "migrate", "api", "litellm"}
