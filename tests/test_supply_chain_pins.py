"""Supply-chain pinning (issue #80): every image `Dockerfile`/`docker-compose.yml` pull is pinned
by digest, not just by a tag that a registry could silently repoint, and dependency installs in
the image build come only from `uv.lock` (`--locked`, no fallback to an unreviewed resolve).

This file also covers `renovate.json`, the update-bot config that keeps those pins (tag *and*
digest) current so they don't quietly rot into a stale, unreviewable reference.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from tests.compose_helpers import (
    docker_compose_available,
    render_compose,
    temporary_env_file,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
COMPOSE_YAML_TEXT = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

DIGEST_RE = re.compile(r"@sha256:[0-9a-f]{64}$")


def _dockerfile_from_lines() -> list[str]:
    """Every `FROM` instruction's image reference, skipping a `FROM <stage>` that refers back to
    an earlier `AS <stage>` build stage rather than pulling an image."""
    stage_names: set[str] = set()
    refs: list[str] = []
    for raw_line in DOCKERFILE.splitlines():
        line = raw_line.strip()
        if not line.upper().startswith("FROM "):
            continue
        parts = line.split()
        # FROM <image> [AS <stage>]
        image_ref = parts[1]
        if image_ref not in stage_names:
            refs.append(image_ref)
        if len(parts) >= 4 and parts[2].upper() == "AS":
            stage_names.add(parts[3])
    return refs


def test_dockerfile_from_instructions_are_digest_pinned() -> None:
    refs = _dockerfile_from_lines()
    assert refs, "no FROM instruction found in Dockerfile"
    for ref in refs:
        assert DIGEST_RE.search(ref), f"FROM image {ref!r} is not pinned with @sha256:<digest>"


def test_dockerfile_uv_sync_uses_locked_with_no_fallback() -> None:
    sync_lines = [line for line in DOCKERFILE.splitlines() if "uv sync" in line]
    assert sync_lines, "no `uv sync` invocation found in Dockerfile"
    for line in sync_lines:
        assert "--locked" in line, f"`uv sync` invocation missing --locked: {line!r}"
        assert "||" not in line, f"`uv sync` invocation still has a `||` fallback: {line!r}"


def _compose_images() -> dict[str, str]:
    data = yaml.safe_load(COMPOSE_YAML_TEXT)
    images = {}
    for name, service in data["services"].items():
        image = service.get("image")
        if image is not None:
            images[name] = image
    return images


def test_compose_images_are_digest_pinned() -> None:
    images = _compose_images()
    # At least postgres and litellm are expected to carry a pulled `image:` reference -- a change
    # that removed both would otherwise make this test vacuously pass.
    assert len(images) >= 2
    for name, image in images.items():
        assert DIGEST_RE.search(image), f"service {name!r} image {image!r} has no @sha256 digest"


@pytest.mark.skipif(
    not docker_compose_available(), reason="docker compose CLI/daemon not available"
)
def test_rendered_compose_images_are_digest_pinned() -> None:
    with temporary_env_file():
        rendered = render_compose()
    checked = 0
    for name, service in rendered["services"].items():
        image = service.get("image")
        if image is None:
            continue
        checked += 1
        assert DIGEST_RE.search(image), f"rendered service {name!r} image {image!r} lacks a digest"
    assert checked >= 2


def _find_pin_digests_true(obj: object) -> bool:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "pinDigests" and value is True:
                return True
            if _find_pin_digests_true(value):
                return True
    elif isinstance(obj, list):
        return any(_find_pin_digests_true(item) for item in obj)
    return False


def test_renovate_json_exists_and_is_valid_json() -> None:
    path = REPO_ROOT / "renovate.json"
    assert path.exists(), "renovate.json is missing from the repo root"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)


def test_renovate_json_enables_digest_pinning_for_docker_managers() -> None:
    data = json.loads((REPO_ROOT / "renovate.json").read_text(encoding="utf-8"))
    assert _find_pin_digests_true(data), "renovate.json does not set pinDigests: true anywhere"

    # The pinDigests rule must actually apply to the Dockerfile/docker-compose managers, not to
    # some unrelated package rule.
    matched_managers: set[str] = set()
    for rule in data.get("packageRules", []):
        if rule.get("pinDigests") is True:
            matched_managers.update(rule.get("matchManagers", []))
    top_level_docker = data.get("docker", {})
    docker_wide_pin = top_level_docker.get("pinDigests") is True
    assert docker_wide_pin or {"dockerfile", "docker-compose"} <= matched_managers


def test_renovate_json_extends_a_recommended_or_best_practices_config() -> None:
    data = json.loads((REPO_ROOT / "renovate.json").read_text(encoding="utf-8"))
    extends = data.get("extends", [])
    assert any("config:recommended" in item or "config:best-practices" in item for item in extends)


def test_renovate_json_keeps_the_uv_lockfile_current() -> None:
    """uv.lock is refreshed by `lockFileMaintenance` (the pep621 manager reads pyproject.toml's
    uv dependencies automatically; the lockfile itself only updates when this is turned on)."""
    data = json.loads((REPO_ROOT / "renovate.json").read_text(encoding="utf-8"))
    lock_file_maintenance = data.get("lockFileMaintenance", {})
    assert lock_file_maintenance.get("enabled") is True


def test_deployment_docs_explain_digest_pins_and_how_to_update_them() -> None:
    deployment_md = (REPO_ROOT / "docs" / "deployment.md").read_text(encoding="utf-8")
    assert "@sha256:" in deployment_md
    assert "imagetools inspect" in deployment_md or "manifest inspect" in deployment_md
    assert "renovate" in deployment_md.lower()
    assert "--locked" in deployment_md


def test_claude_md_notes_digest_pins_and_locked_installs() -> None:
    claude_md = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "digest" in claude_md.lower()
    assert "--locked" in claude_md
    assert "docs/deployment.md" in claude_md
