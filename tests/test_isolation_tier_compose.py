"""S10-T6 (#78): the rendered `docker-compose.yml` carries no second database service or
dedicated-tenant credential anywhere, so the hybrid-isolation seam (ADR-0002) stays zero-footprint
until an operator actually provisions a dedicated tenant. Skipped when Docker is unavailable.

Rendering is shared with the other compose tests via tests/compose_helpers.py."""

from __future__ import annotations

import pytest

from tests.compose_helpers import docker_compose_available, render_compose, temporary_env_file

# The one file this template ships by default under a secrets-style mount: the Postgres init
# script (a config file, not a per-tenant credential). Anything else that looks like it lives
# under a secrets directory is treated as a dedicated-tenant secret file for this test's purposes.
_SHIPPED_SECRET_LIKE_MOUNTS = {"./docker/postgres/01-init.sh"}

# The alias directory app/db/engine_registry.py reads dedicated-tenant DSNs from — must never be
# mounted into any service by default (no dedicated tenant exists in the shipped template).
_TENANT_DB_SECRETS_MARKER = "tenant-db"


pytestmark = pytest.mark.skipif(
    not docker_compose_available(), reason="docker compose CLI/daemon not available"
)


@pytest.fixture
def rendered_config():
    with temporary_env_file():
        return render_compose()


def _is_postgres_image(image: str | None) -> bool:
    if not image:
        return False
    lowered = image.lower()
    # The template ships `pgvector/pgvector:pg17` (a Postgres image with the pgvector
    # extension baked in), not the bare `postgres` image — match both.
    return "postgres" in lowered or "pgvector" in lowered


def test_exactly_one_postgres_service(rendered_config):
    postgres_services = [
        name
        for name, service in rendered_config["services"].items()
        if _is_postgres_image(service.get("image"))
    ]
    assert postgres_services == ["postgres"], (
        f"expected exactly one Postgres service, found: {postgres_services}"
    )


def test_no_dedicated_tenant_connection_string_or_alias_in_any_environment(rendered_config):
    for name, service in rendered_config["services"].items():
        env = service.get("environment") or {}
        # docker compose config normalizes environment to a dict or a list of "KEY=VALUE".
        items = env.items() if isinstance(env, dict) else (e.split("=", 1) for e in env if "=" in e)
        for key, value in items:
            haystack = f"{key}={value}".lower()
            assert _TENANT_DB_SECRETS_MARKER not in haystack, (
                f"service {name!r} environment references a dedicated-tenant secret path: "
                f"{key}={value}"
            )
            assert "database_alias" not in haystack and "db_alias" not in haystack, (
                f"service {name!r} environment names a database alias: {key}={value}"
            )


def test_no_service_mounts_a_dedicated_tenant_secret_file(rendered_config):
    for name, service in rendered_config["services"].items():
        for volume in service.get("volumes") or []:
            source = volume.get("source") if isinstance(volume, dict) else str(volume).split(":")[0]
            if source is None:
                continue
            source_str = str(source)
            if source_str in _SHIPPED_SECRET_LIKE_MOUNTS:
                continue
            assert _TENANT_DB_SECRETS_MARKER not in source_str.lower(), (
                f"service {name!r} mounts a dedicated-tenant secret path not shipped by "
                f"default: {source_str}"
            )
