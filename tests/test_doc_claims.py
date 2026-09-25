"""The template's own documentation claims about roles, isolation, and residency (issue #19,
Spec 1's closing ticket).

The 2026-09-12 security review (`docs/reviews/2026-09-12-security-review.md`) found three
already-false claims in the checked-in README/CLAUDE.md/docs: that the api container "never
holds the superuser DSN" (docs/deployment.md), that tools "run with the logged-in user's
permissions" (README.md, implying per-member isolation the code has never had), and that "GDPR
is covered via EU regions + DPA" (README.md, stated as delivered when residency routing is still
an open decision, ADR-0008). This test asserts the retired wording is gone from the documentation
files and each one's corrected replacement is present, so a later edit cannot silently
reintroduce a claim the code does not back up.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
CLAUDE_MD = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
DEPLOYMENT_MD = (REPO_ROOT / "docs" / "deployment.md").read_text(encoding="utf-8")
ADR_0011 = (REPO_ROOT / "docs" / "adr" / "0011-control-plane-and-secrets.md").read_text(
    encoding="utf-8"
)

RETIRED_CLAIMS = {
    "superuser-DSN wording": [
        "the api container never holds the superuser DSN",
        "keeps the superuser DSN out of the long-running api container",
    ],
    "per-user-permissions wording": [
        "tools that run with the logged-in user's permissions",
    ],
    "residency-covered-by-EU-regions wording": [
        "GDPR is covered via EU regions + DPA",
    ],
}

ALL_DOCS_TEXT = "\n".join([README, CLAUDE_MD, DEPLOYMENT_MD])


def test_retired_superuser_dsn_wording_is_gone() -> None:
    for phrase in RETIRED_CLAIMS["superuser-DSN wording"]:
        assert phrase not in ALL_DOCS_TEXT, f"retired claim still present: {phrase!r}"


def test_retired_per_user_permissions_wording_is_gone() -> None:
    for phrase in RETIRED_CLAIMS["per-user-permissions wording"]:
        assert phrase not in ALL_DOCS_TEXT, f"retired claim still present: {phrase!r}"


def test_retired_residency_eu_regions_wording_is_gone() -> None:
    for phrase in RETIRED_CLAIMS["residency-covered-by-EU-regions wording"]:
        assert phrase not in ALL_DOCS_TEXT, f"retired claim still present: {phrase!r}"


def test_readme_states_isolation_is_by_tenant_not_by_member() -> None:
    assert "isolated by tenant, not by individual member" in README
    assert "per-member visibility is not implemented" in README


def test_readme_states_residency_is_an_open_decision() -> None:
    assert "not yet implemented" in README
    assert "ADR-0008" in README


def test_deployment_docs_describe_the_actual_env_allow_list() -> None:
    # The corrected replacement names the actual, narrower allow-list instead of the retired
    # "never holds the superuser DSN" phrasing.
    assert "app_owner" in DEPLOYMENT_MD
    assert "DATABASE_URL_MIGRATIONS" in DEPLOYMENT_MD
    assert "APP_OWNER_DB_PASSWORD" in DEPLOYMENT_MD


def test_claude_md_do_not_touch_list_names_role_bootstrap_and_control_plane_migration() -> None:
    section = CLAUDE_MD.split("## Do not touch without checking first", 1)[1]
    assert "01-init.sh" in section
    assert "provision_roles.py" in section
    assert "0002_control_plane_schema.py" in section


def test_adr_0011_is_accepted_not_proposed() -> None:
    status_line = next(line for line in ADR_0011.splitlines() if line.startswith("- **Status:**"))
    assert "accepted" in status_line
    assert "proposed" not in status_line
