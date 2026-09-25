"""Rendered `docker-compose.yml` configuration (Spec 9 / #67): the standalone
`scripts/provision_roles.py` role-provisioning script must not be wired into the compose stack's
automatic startup — an operator on managed Postgres runs it by hand, once, outside compose
entirely. Skipped when the Docker CLI/daemon isn't available.

Rendering itself (Docker availability check, throwaway `.env`, `docker compose config`) is
shared with the other compose tests via tests/compose_helpers.py (Spec 1 / #18).
"""

from __future__ import annotations

import pytest

from tests.compose_helpers import docker_compose_available, render_compose, temporary_env_file

pytestmark = pytest.mark.skipif(
    not docker_compose_available(), reason="docker compose CLI/daemon not available"
)


@pytest.fixture
def rendered_config():
    with temporary_env_file():
        return render_compose()


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
