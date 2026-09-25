"""Rendered `docker-compose.yml` configuration (Spec 8 / #64): the trace sink is a
content-bearing, residency-governed endpoint exactly like the database and gateway hosts
(ADR-0008) — it must reach the two services that need it (`api`, which runs agent traffic, and
`migrate`, which runs against the same residency-governed deployment) and must never reach a
service that has no business receiving it (`postgres`, `litellm`).

`api` and `migrate` both declare `env_file: .env` in docker-compose.yml, which is what actually
carries `LANGFUSE_HOST`/`RESIDENCY` to them; `postgres` and `litellm` only ever get the specific
variables named under their own `environment:` block. This test renders the real compose file
and asserts that wiring directly, so an edit that removes `env_file: .env` from a service that
needs the trace sink — or that starts forwarding it to a service that shouldn't have it — fails
here instead of in an incident.

A minimal render helper is duplicated here rather than imported from
tests/test_compose_config.py or tests/test_gateway_compose.py on purpose (#18 consolidates the
render fixtures across all three files); skipped cleanly when the Docker CLI/daemon isn't
available.
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

# The residency- and tracing-related variables this test tracks (ADR-0008): the deployment's own
# residency, and the trace sink's coordinates (host, public key, secret key).
RESIDENCY_VAR = "RESIDENCY"
TRACE_SINK_VARS = ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")

# Services that run against tenant content and therefore need the trace sink reachable.
SERVICES_THAT_NEED_THE_TRACE_SINK = ("api", "migrate")
# Services with no business receiving it: the database itself, and the model gateway (which gets
# only its own database URL, its master key, and model-provider credentials — never the tracing
# secret; see tests/test_gateway_compose.py).
SERVICES_WITH_NO_BUSINESS_RECEIVING_IT = ("postgres", "litellm")

TEST_LANGFUSE_HOST = "https://eu.cloud.langfuse.com"
TEST_RESIDENCY = "eu"


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
def env_file():
    """A throwaway `.env`, with the residency and trace-sink variables filled in, so
    `env_file: .env` on the api/migrate services resolves to real values this test can assert
    on. Written into the real project directory only for the duration of the test, restored (or
    removed, if it didn't exist before) afterward — never left behind."""
    real_env = REPO_ROOT / ".env"
    existed = real_env.exists()
    backup = real_env.read_text() if existed else None
    example = (REPO_ROOT / ".env.example").read_text()
    example = example.replace("LANGFUSE_HOST=", f"LANGFUSE_HOST={TEST_LANGFUSE_HOST}")
    example = example.replace("RESIDENCY=eu", f"RESIDENCY={TEST_RESIDENCY}")
    real_env.write_text(example)
    try:
        yield real_env
    finally:
        if existed:
            real_env.write_text(backup)
        else:
            real_env.unlink(missing_ok=True)


def _render(env_file) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "LITELLM_MASTER_KEY"}
    env["LITELLM_MASTER_KEY"] = "test-master-key"
    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "config", "--format", "json"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("service_name", SERVICES_THAT_NEED_THE_TRACE_SINK)
def test_trace_sink_and_residency_reach_services_that_need_them(env_file, service_name):
    rendered = _render(env_file)
    service_env = rendered["services"][service_name].get("environment", {})

    assert service_env.get("LANGFUSE_HOST") == TEST_LANGFUSE_HOST, (
        f"service {service_name!r} must receive the trace-sink host — it processes tenant "
        "content and its traces must land in the residency's trace sink (ADR-0008)"
    )
    assert service_env.get("RESIDENCY") == TEST_RESIDENCY, (
        f"service {service_name!r} must receive the deployment's residency setting"
    )


@pytest.mark.parametrize("service_name", SERVICES_WITH_NO_BUSINESS_RECEIVING_IT)
def test_trace_sink_is_absent_from_services_with_no_business_receiving_it(env_file, service_name):
    rendered = _render(env_file)
    service_env = rendered["services"][service_name].get("environment", {})

    for var in TRACE_SINK_VARS:
        assert var not in service_env, (
            f"service {service_name!r} must never receive {var} — it has no business "
            "processing tenant content or exporting traces"
        )


def test_known_services_partition_completely(env_file):
    """Every service in the compose file is accounted for above — so a new service added later
    without an explicit residency/tracing decision is caught here instead of silently falling
    into neither bucket."""
    rendered = _render(env_file)
    accounted_for = set(SERVICES_THAT_NEED_THE_TRACE_SINK) | set(
        SERVICES_WITH_NO_BUSINESS_RECEIVING_IT
    )
    assert set(rendered["services"]) == accounted_for
