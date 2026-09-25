"""Shared plumbing for rendering `docker-compose.yml` via `docker compose config` in tests
(Spec 1 / #18): every compose test used to carry its own copy of the "is Docker available",
"write a throwaway .env" and "run `docker compose config --format json`" logic. This module is
the one place that logic lives now; test_compose_config.py, test_gateway_compose.py,
test_residency_trace_sink_compose.py, and test_deployment_hardening_compose.py all import it
instead of re-implementing it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"


def docker_compose_available() -> bool:
    """True if the Docker CLI/daemon can actually render a compose file (some CI runners have
    neither)."""
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "compose", "version"], capture_output=True, check=True, timeout=10
        )
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return False
    return True


@contextmanager
def temporary_env_file(overrides: dict[str, str] | None = None) -> Iterator[Path]:
    """Writes a throwaway `.env` into the repo root for the duration of the `with` block —
    `env_file: .env` on the api/migrate services is a fixed reference in the compose file
    itself, so `docker compose --env-file` alone does not satisfy it; a real `.env` must exist
    on disk. Built from `.env.example`, with `overrides` applied as `KEY=` -> `KEY=value`
    replacements on top of the template's own line for that key. Restores the real `.env` (or
    removes the throwaway one) afterward — never left behind, and never overwritten if the repo
    already has a real `.env`.
    """
    real_env = REPO_ROOT / ".env"
    existed = real_env.exists()
    backup = real_env.read_text() if existed else None
    content = (REPO_ROOT / ".env.example").read_text()
    for key, value in (overrides or {}).items():
        content, count = re.subn(rf"^{re.escape(key)}=.*$", f"{key}={value}", content, flags=re.M)
        if count == 0:
            content += f"\n{key}={value}\n"
    real_env.write_text(content)
    try:
        yield real_env
    finally:
        if existed:
            real_env.write_text(backup)
        else:
            real_env.unlink(missing_ok=True)


def render_compose(extra_env: dict[str, str | None] | None = None) -> dict:
    """Renders `docker-compose.yml` with `docker compose config --format json`. Requires a real
    `.env` on disk already (see `temporary_env_file`). `LITELLM_MASTER_KEY` defaults to a test
    value; pass `{"LITELLM_MASTER_KEY": None}` in `extra_env` to unset it entirely (rather than
    just overriding it to an empty string), e.g. to exercise the required-variable failure."""
    env = {k: v for k, v in os.environ.items() if k != "LITELLM_MASTER_KEY"}
    env["LITELLM_MASTER_KEY"] = "test-master-key"
    for key, value in (extra_env or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
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
