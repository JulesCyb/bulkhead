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

from app.config import RESIDENCY_ALLOW_LIST, Settings
from app.tenant_settings import TenantSettings

REPO_ROOT = Path(__file__).resolve().parent.parent

README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
CLAUDE_MD = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
DEPLOYMENT_MD = (REPO_ROOT / "docs" / "deployment.md").read_text(encoding="utf-8")
CONTEXT_MD = (REPO_ROOT / "CONTEXT.md").read_text(encoding="utf-8")
ENV_EXAMPLE = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
ADR_0011 = (REPO_ROOT / "docs" / "adr" / "0011-control-plane-and-secrets.md").read_text(
    encoding="utf-8"
)
ADR_0002 = (REPO_ROOT / "docs" / "adr" / "0002-hybrid-tenant-isolation.md").read_text(
    encoding="utf-8"
)
ADR_0009 = (REPO_ROOT / "docs" / "adr" / "0009-cost-and-abuse-protection.md").read_text(
    encoding="utf-8"
)
ADR_DIR = REPO_ROOT / "docs" / "adr"

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


def test_readme_states_residency_is_delivered_not_an_open_decision() -> None:
    # Spec 8's closing ticket (#63): residency routing (#58, #59, #60, #61, #62) landed, so the
    # README's "not yet implemented" wording from Spec 1 (#19) is now itself a false claim -- the
    # opposite problem this file exists to catch. It is replaced with a pointer at the real,
    # testable claim in docs/residency.md, and ADR-0008 is cited as accepted.
    assert "not yet implemented" not in README
    assert "ADR-0008" in README
    assert "docs/residency.md" in README


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


# Spec 10's closing ticket (#79): ADR-0002 moves from proposed to accepted now that the seam it
# describes (engine-per-tenant resolution, per-alias migrations, the runtime alias guard) is the
# one actually built, and the contributor guide / onboarding docs describe that behaviour instead
# of a second, drifting copy of it.


def test_adr_0002_is_accepted_not_proposed() -> None:
    status_line = next(line for line in ADR_0002.splitlines() if line.startswith("- **Status:**"))
    assert "accepted" in status_line
    assert "proposed" not in status_line


def test_claude_md_db_access_rule_describes_internal_engine_resolution() -> None:
    rule_3 = next(
        line
        for line in CLAUDE_MD.splitlines()
        if line.strip().startswith("3. **DB access only through repositories**")
    )
    section_start = CLAUDE_MD.index(rule_3)
    section_end = CLAUDE_MD.index("\n4. ", section_start)
    section = " ".join(CLAUDE_MD[section_start:section_end].split())

    assert "resolves" in section
    assert "isolation tier" in section
    assert "database alias" in section
    assert "pooled by default" in section
    assert "no change to how callers use it" in section


def test_claude_md_do_not_touch_list_names_engine_registry_with_session_layer() -> None:
    section = CLAUDE_MD.split("## Do not touch without checking first", 1)[1]
    section = section.split("## Agent skills", 1)[0]

    assert "app/db/engine_registry.py" in section
    session_index = section.index("app/db/session.py")
    engine_registry_index = section.index("app/db/engine_registry.py")
    # The engine-registry bullet is the very next bullet after the session-layer entry (no
    # unrelated bullet — such as the append-only control-plane tables — sits between them).
    assert session_index < engine_registry_index
    between = section[session_index:engine_registry_index]
    assert between.count("\n- ") <= 1


def test_readme_onboarding_table_tenant_session_row_notes_engine_resolution() -> None:
    row = next(line for line in README.splitlines() if line.startswith("| Tenant session |"))
    assert "database" in row
    assert "pooled by default" in row
    assert "ADR-0002" in row
    # It points at the ADR's trigger list rather than restating it: none of the concrete
    # trigger phrases from ADR-0002's "Revisit when" section appear a second time here.
    assert "contractually" not in row
    assert "regulated data" not in row


def test_no_duplicate_dedicated_tenant_trigger_list_outside_adr_0002() -> None:
    # ADR-0002's own "Revisit when" list is the one place naming the concrete triggers for
    # provisioning a dedicated tenant. No other doc should restate that list.
    trigger_phrases = [
        "a customer demands a dedicated database",
        "regulated data (health, finance, public sector) enters a tenant",
        "one tenant dominates load or size",
        "needs their own backup, restore, or export cadence",
    ]
    other_docs = {
        "README.md": README,
        "CLAUDE.md": CLAUDE_MD,
        "CONTEXT.md": CONTEXT_MD,
        "docs/deployment.md": DEPLOYMENT_MD,
    }
    for adr_path in ADR_DIR.glob("*.md"):
        if adr_path.name == "0002-hybrid-tenant-isolation.md":
            continue
        other_docs[f"docs/adr/{adr_path.name}"] = adr_path.read_text(encoding="utf-8")

    for name, text in other_docs.items():
        for phrase in trigger_phrases:
            assert phrase not in text, f"duplicate trigger list found in {name}: {phrase!r}"


# Spec 7's closing ticket (#57): the gateway stopped being an optional profile once #51-#56
# landed (ADR-0009). This block asserts the documentation's own description of the stack, the
# module table, and the deployment env vars all say so -- no doc left telling a reader to opt in
# with a profile flag -- and that ADR-0009 itself records "accepted" now that the whole spec has
# landed.

RETIRED_OPTIONAL_GATEWAY_PHRASES = [
    "optionally through the LiteLLM gateway",
    "Optional: route everything through the LiteLLM gateway",
    "provider abstraction, LiteLLM option",
]

ALL_DOCS_AND_ENV_TEXT = "\n".join([README, CLAUDE_MD, DEPLOYMENT_MD, CONTEXT_MD, ENV_EXAMPLE])


def test_retired_optional_gateway_wording_is_gone() -> None:
    for phrase in RETIRED_OPTIONAL_GATEWAY_PHRASES:
        assert phrase not in ALL_DOCS_AND_ENV_TEXT, f"retired claim still present: {phrase!r}"


def test_claude_md_stack_description_states_gateway_is_mandatory() -> None:
    start = next(
        i for i, line in enumerate(CLAUDE_MD.splitlines()) if line.strip().startswith("- Models:")
    )
    lines = CLAUDE_MD.splitlines()
    end = next(i for i in range(start + 1, len(lines)) if not lines[i].startswith("  "))
    stack_block = " ".join(lines[start:end])
    assert "always through the LiteLLM gateway" in stack_block
    assert "ADR-0009" in stack_block


def test_readme_models_row_states_gateway_is_mandatory() -> None:
    row = next(line for line in README.splitlines() if line.startswith("| Models |"))
    assert "mandatory" in row
    assert "ADR-0009" in row


def test_env_example_gateway_vars_are_documented_as_required() -> None:
    assert "Required (ADR-0009): every model/embedding call goes through the LiteLLM gateway" in (
        ENV_EXAMPLE
    )


def test_context_md_states_budget_stops_only_the_exhausted_tenant() -> None:
    # In the project's own domain terms (CONTEXT.md "Budget"): an exhausted budget stops the
    # tenant that exhausted it, never the whole deployment.
    budget_section = " ".join(
        CONTEXT_MD.split("**Budget**:", 1)[1].split("**Run limit**:", 1)[0].split()
    )
    assert "An exhausted budget stops that tenant, never the deployment" in budget_section


def test_context_md_states_what_a_run_limit_protects_against_that_a_budget_does_not() -> None:
    # A run limit guards against loops (including a poisoned-document-provoked one) inside a
    # single run -- a failure mode a spend budget, measured after the fact, does not catch.
    run_limit_section = " ".join(
        CONTEXT_MD.split("**Run limit**:", 1)[1].split("**Operator**:", 1)[0].split()
    )
    assert "protects against loops" in run_limit_section
    assert "not against cost" in run_limit_section


def test_adr_0009_is_accepted_not_proposed() -> None:
    status_line = next(line for line in ADR_0009.splitlines() if line.startswith("- **Status:**"))
    assert "accepted" in status_line
    assert "proposed" not in status_line


# Spec 3's closing ticket (#30 / S3-T5): the claim that roles only gate actions and never hide
# content is now backed by a test (tests/test_rls_integration.py), and ADR-0004 -- the decision
# record for this authorization model -- is corrected from "exactly three roles" to the real
# four-role model and moves from proposed to accepted.

ADR_0004 = (
    REPO_ROOT / "docs" / "adr" / "0004-authorization-tenant-in-db-roles-in-app.md"
).read_text(encoding="utf-8")
ADR_0005 = (REPO_ROOT / "docs" / "adr" / "0005-agent-identities.md").read_text(encoding="utf-8")
ARCHITECTURE_SVG = (REPO_ROOT / "docs" / "architecture.svg").read_text(encoding="utf-8")


def test_adr_0004_is_accepted_not_proposed() -> None:
    status_line = next(line for line in ADR_0004.splitlines() if line.startswith("- **Status:**"))
    assert "accepted" in status_line
    assert "proposed" not in status_line


def test_adr_0004_names_all_four_roles_and_cross_references_adr_0005() -> None:
    assert "exactly three roles" not in ADR_0004
    decision_section = " ".join(
        ADR_0004.split("## Decision", 1)[1].split("## Consequences", 1)[0].split()
    )
    for role in ("admin", "member", "support", "agent"):
        assert f"`{role}`" in decision_section
    assert "ADR-0005" in decision_section


def test_architecture_svg_caption_matches_tenant_wide_visibility() -> None:
    assert "the logged-in user's permissions" not in ARCHITECTURE_SVG
    assert "Tenant-wide visibility, roles gate actions" in ARCHITECTURE_SVG


# Spec 8's closing ticket (#63): the residency/tracing claim in the outward-facing docs, the
# sub-processor list, and the worked "add a second residency" recipe are checked directly against
# the Settings/allow-list objects the code actually enforces, not against a second copy of the
# same prose -- so a change to RESIDENCY_ALLOW_LIST, Settings, or TenantSettings that isn't
# reflected in docs/residency.md fails this file instead of only being caught by a human review.

RESIDENCY_MD = (REPO_ROOT / "docs" / "residency.md").read_text(encoding="utf-8")
ADR_0008 = (REPO_ROOT / "docs" / "adr" / "0008-residency-per-tenant.md").read_text(encoding="utf-8")
MCP_SERVER_SOURCE = (REPO_ROOT / "app" / "mcp" / "server.py").read_text(encoding="utf-8")


def test_adr_0008_is_accepted_not_proposed() -> None:
    status_line = next(line for line in ADR_0008.splitlines() if line.startswith("- **Status:**"))
    assert "accepted" in status_line
    assert "proposed" not in status_line


def test_residency_doc_claim_names_what_the_code_actually_enforces() -> None:
    # Each named guarantee cross-checked against the object the running code builds, not just
    # against itself as a string.
    assert "per-tenant residency setting" in RESIDENCY_MD
    assert "control.tenants.residency" in RESIDENCY_MD

    assert "startup allow-list check" in RESIDENCY_MD
    assert "run_startup_checks" in RESIDENCY_MD

    # "No default embedding provider": the doc's claim is only true if Settings really has no
    # default for either field.
    assert Settings.model_fields["embedding_provider"].default is None
    assert Settings.model_fields["embedding_model"].default is None
    assert "no default embedding provider" in RESIDENCY_MD.lower()

    # "Content-free tracing by default": true only if TenantSettings really defaults the opt-in
    # flag to False.
    assert TenantSettings.model_fields["content_tracing_opt_in"].default is False
    assert "content_tracing_opt_in" in RESIDENCY_MD
    assert "default `False`" in RESIDENCY_MD


def test_residency_doc_sub_processor_list_matches_the_allow_list_object() -> None:
    # Every host RESIDENCY_ALLOW_LIST actually contains is named in the doc -- not a hand-copied
    # second list that could silently drift from it.
    for residency, route in RESIDENCY_ALLOW_LIST.items():
        assert residency in RESIDENCY_MD, f"residency {residency!r} not named in docs/residency.md"
        assert route.trace_sink_host in RESIDENCY_MD, (
            f"trace sink host {route.trace_sink_host!r} for {residency!r} not named in the doc"
        )
        embedding_host = route.embedding_endpoint.split("//", 1)[-1].split("/", 1)[0]
        assert embedding_host in RESIDENCY_MD, (
            f"embedding endpoint host {embedding_host!r} for {residency!r} not named in the doc"
        )
        for pattern in route.model_host_patterns:
            bare_domain = pattern.lstrip("*.")
            assert bare_domain in RESIDENCY_MD, (
                f"model host domain {bare_domain!r} for {residency!r} not named in the doc"
            )


def test_residency_doc_has_a_worked_second_residency_recipe() -> None:
    section = RESIDENCY_MD.split("## Worked example: adding a second residency", 1)[1]
    section = section.split("## The MCP boundary", 1)[0]
    assert "RESIDENCY_ALLOW_LIST" in section
    assert "RESIDENCY_MODEL_ALLOW_LIST" in section
    assert "docker/litellm/config.yaml" in section
    assert "trace_sink_host" in section


def test_mcp_server_docs_state_the_client_model_boundary_and_snippet_cap() -> None:
    normalized = " ".join(MCP_SERVER_SOURCE.split())
    assert "outside this application's processor chain" in normalized
    assert "outside residency enforcement" in normalized
    assert "own model" in normalized
    # The existing cap this boundary statement names as the bound that still applies.
    assert "limit = max(1, min(limit, 20))" in MCP_SERVER_SOURCE


def test_residency_doc_names_mcp_over_http_as_still_open() -> None:
    section = RESIDENCY_MD.split("## The MCP boundary", 1)[1]
    assert "#49" in section
    assert "stdio" in section


def test_claude_md_tracing_rule_describes_flat_attributes_and_content_off_by_default() -> None:
    rule_7 = next(line for line in CLAUDE_MD.splitlines() if line.strip().startswith("7. **Every"))
    section_start = CLAUDE_MD.index(rule_7)
    section_end = CLAUDE_MD.index("\n8. ", section_start)
    section = " ".join(CLAUDE_MD[section_start:section_end].split())

    assert "flat" in section
    assert "content_tracing_opt_in" in section
    assert "default `False`" in section
    assert "as `metadata`" not in section or "flat" in section  # flat attributes, not only metadata


def test_claude_md_model_rule_describes_residency_resolution() -> None:
    rule_6 = next(line for line in CLAUDE_MD.splitlines() if line.strip().startswith("6. **Models"))
    section_start = CLAUDE_MD.index(rule_6)
    section_end = CLAUDE_MD.index("\n7. ", section_start)
    section = " ".join(CLAUDE_MD[section_start:section_end].split())

    assert "resolve_residency_route" in section
    assert "ResidencyUnresolved" in section
    assert "run_startup_checks" in section
    assert "no default embedding provider" in section.lower()


def test_tenant_settings_catalog_documents_residency_and_content_tracing_opt_in() -> None:
    import app.tenant_settings as tenant_settings_module

    doc = tenant_settings_module.__doc__ or ""
    assert "## Catalog" in doc
    assert "residency" in doc
    assert "control.tenants.residency" in doc
    assert "content_tracing_opt_in" in doc
    # Shape, validation, and default are all named for both entries, not just one.
    assert "RESIDENCY_ALLOW_LIST" in doc
    assert "no default" in doc
    assert "False" in doc


def test_claude_md_audit_rule_points_at_documents_columns_not_aspirational() -> None:
    rule_1b = next(line for line in CLAUDE_MD.splitlines() if line.strip().startswith("1b. **The"))
    section_start = CLAUDE_MD.index(rule_1b)
    section_end = CLAUDE_MD.index("\n2. ", section_start)
    section = " ".join(CLAUDE_MD[section_start:section_end].split())

    assert "app.identity_id" in section
    assert "documents.created_by" in section
    assert "documents.updated_by" in section
    assert "0010_document_audit_columns.py" in section
    for aspirational_word in ("will", "would", "eventually", "once implemented"):
        assert aspirational_word not in section
