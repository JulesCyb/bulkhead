"""Rendered `docker-compose.yml` configuration (Spec 1 / #18 — deployment hardening): every
service receives only the environment variables its own job needs, every pulled image is pinned
to a specific version, and the gateway shares no network with the application/migration
services.

Rendering itself (Docker availability check, throwaway `.env`, `docker compose config`) is
shared with the other compose tests via tests/compose_helpers.py.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from tests.compose_helpers import (
    REPO_ROOT,
    docker_compose_available,
    render_compose,
    temporary_env_file,
)

pytestmark = pytest.mark.skipif(
    not docker_compose_available(), reason="docker compose CLI/daemon not available"
)

# The database owner's and cluster superuser's own secrets/connection strings: never the
# long-running api container's job.
DATABASE_OWNER_SECRETS = {
    "DATABASE_URL_MIGRATIONS",
    "APP_OWNER_DB_PASSWORD",
    "POSTGRES_PASSWORD",
}

# Locally-created env-var-file variants an operator might create by hand; every one of them must
# be excluded from version control and the image build context by default, unlike the tracked
# example template.
LOCAL_ENV_FILE_VARIANTS = [".env.local", ".env.production", ".env.staging"]


@pytest.fixture
def env_file():
    with temporary_env_file() as path:
        yield path


def test_api_service_has_no_database_owner_credentials(env_file):
    rendered = render_compose()
    api_env = rendered["services"]["api"]["environment"]
    assert set(api_env) & DATABASE_OWNER_SECRETS == set()
    assert "app_owner" not in api_env.get("DATABASE_URL", "")


def test_migrate_service_receives_only_migration_and_tracing_variables(env_file):
    """The migration job is one-shot and needs the owner DSN plus the trace-sink/residency
    wiring it shares with `api` (tests/test_residency_trace_sink_compose.py) — nothing else: no
    gateway master key, no model/embedding provider credentials, no application role password."""
    rendered = render_compose()
    migrate_env = rendered["services"]["migrate"]["environment"]

    forbidden = {
        "LITELLM_MASTER_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "APP_DB_PASSWORD",
        "GATEWAY_DB_PASSWORD",
        "POSTGRES_PASSWORD",
    }
    assert set(migrate_env) & forbidden == set()
    assert "DATABASE_URL_MIGRATIONS" in migrate_env


def test_gateway_has_no_route_to_the_applications_own_database_traffic(env_file):
    """The gateway must not sit on `db`, the network `api`/`migrate` use to reach postgres for
    the application's own tenant data — a compromised gateway credential then has no network
    route to that traffic, on top of the role-level isolation in docker/postgres/01-init.sh.

    Deviation from a fully literal reading of the acceptance criterion ("the database service and
    the model gateway service are not attached to a common network"): the gateway's own database
    lives on the same Postgres instance (ADR-0009, #51 — out of this ticket's scope to redesign),
    so `postgres` and `litellm` still share a dedicated `gatewaydb` network carrying only the
    gateway's own database traffic (verified separately below); a Postgres bound to
    `127.0.0.1` on the host cannot be reached from a container with no shared Docker network
    (verified empirically), so full network-level separation from `postgres` itself is not
    achievable without either unbinding that port (docs/deployment.md says not to) or splitting
    the gateway onto its own Postgres instance (a bigger change than this ticket's scope). What is
    achievable, and enforced here, is that the gateway has no network route to `api`'s or
    `migrate`'s own database traffic, or to `migrate` at all."""
    rendered = render_compose()
    services = rendered["services"]

    def networks_of(name: str) -> set[str]:
        nets = services[name].get("networks")
        if nets is None:
            return {"default"}
        return set(nets)

    gateway_networks = networks_of("litellm")
    assert "db" not in gateway_networks
    assert gateway_networks & networks_of("migrate") == set()


def test_gateway_and_postgres_database_traffic_is_on_its_own_dedicated_network(env_file):
    """The gateway's own database traffic to postgres (DATABASE_URL) is confined to a network
    (`gatewaydb`) that neither `api` nor `migrate` is a member of — so even though `postgres` and
    `litellm` do share a network (see the deviation note above), the application's own database
    connections are not reachable over it."""
    rendered = render_compose()
    services = rendered["services"]

    def networks_of(name: str) -> set[str]:
        nets = services[name].get("networks")
        if nets is None:
            return {"default"}
        return set(nets)

    shared = networks_of("postgres") & networks_of("litellm")
    assert shared, "litellm has no network path to postgres for its own database at all"
    for service_name in ("api", "migrate"):
        assert shared & networks_of(service_name) == set(), (
            f"{service_name!r} shares the gateway's own database network {shared!r}"
        )


def test_every_pulled_image_reference_is_pinned_to_a_specific_version(env_file):
    """Every service with an `image:` (a pulled, third-party image — `api`/`migrate` are built
    locally from this repo's own Dockerfile and have no such reference) must name a specific,
    non-floating version, not `latest`/`main`/`stable`."""
    rendered = render_compose()
    floating_tags = {"latest", "main", "main-stable", "stable"}

    checked = 0
    for name, service in rendered["services"].items():
        image = service.get("image")
        if image is None:
            continue
        checked += 1
        tag = image.rsplit(":", 1)[-1]
        assert tag not in floating_tags, f"service {name!r} uses a floating image tag {tag!r}"
        assert any(char.isdigit() for char in tag), (
            f"service {name!r} image tag {tag!r} does not name a specific version"
        )
    # At least postgres and litellm are expected to carry a pulled `image:` reference — a change
    # that removes both `image:` keys (e.g. by switching everything to `build:`) would otherwise
    # make this test vacuously pass.
    assert checked >= 2


def test_env_file_variants_are_git_ignored():
    """A locally created `.env.<anything>` must be excluded from version control by default,
    while the tracked example template stays tracked."""
    for name in LOCAL_ENV_FILE_VARIANTS:
        result = subprocess.run(["git", "check-ignore", "-q", name], cwd=REPO_ROOT)
        assert result.returncode == 0, f"{name} is not ignored by .gitignore"

    result = subprocess.run(["git", "check-ignore", "-q", ".env.example"], cwd=REPO_ROOT)
    assert result.returncode == 1, ".env.example must stay tracked, not ignored"


def test_env_file_variants_are_excluded_from_the_image_build_context(tmp_path):
    """.dockerignore uses gitignore-compatible pattern syntax (Docker's own documentation);
    mirroring it into a throwaway git repo evaluates the same rules without needing an actual
    image build."""
    dockerignore = (REPO_ROOT / ".dockerignore").read_text()
    (tmp_path / ".gitignore").write_text(dockerignore)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    for name in [*LOCAL_ENV_FILE_VARIANTS, ".env.example"]:
        (tmp_path / name).write_text("")

    for name in LOCAL_ENV_FILE_VARIANTS:
        result = subprocess.run(["git", "check-ignore", "-q", name], cwd=tmp_path)
        assert result.returncode == 0, f"{name} is not excluded from the image build context"

    result = subprocess.run(["git", "check-ignore", "-q", ".env.example"], cwd=tmp_path)
    assert result.returncode == 1, ".env.example must stay in the image build context"


def test_secrets_dir_documented_in_deployment_docs():
    """The file-backed secrets path for production (issue #14 / ADR-0011, `Settings.model_config
    secrets_dir="/run/secrets"`) must be documented for operators, not just implemented."""
    deployment_docs = (REPO_ROOT / "docs" / "deployment.md").read_text()
    assert "/run/secrets" in deployment_docs
    assert re.search(r"secrets_dir", deployment_docs)
