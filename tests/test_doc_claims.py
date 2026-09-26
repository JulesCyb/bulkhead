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

import json
import os
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.residency import ResidencyAllowList
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


# ai-app-starter#7 review finding: the gateway being "mandatory" used to be documentation only --
# `Settings.litellm_base_url` was optional and nothing enforced it at startup. This block asserts
# the docs now say enforcement is real (Settings construction + startup_checks), not just intent,
# and that Settings actually behaves that way -- the doc claim and the code are checked together.


def test_claude_md_states_gateway_is_enforced_at_startup() -> None:
    assert "Enforced at startup" in CLAUDE_MD
    assert "`Settings` refuses to construct" in CLAUDE_MD


def test_readme_models_row_states_startup_enforcement() -> None:
    row = next(line for line in README.splitlines() if line.startswith("| Models |"))
    assert "enforced at startup" in row


def test_deployment_md_states_gateway_enforcement_is_not_only_documented() -> None:
    assert "Enforced at startup, not only documented" in DEPLOYMENT_MD


def test_settings_actually_refuses_to_construct_without_a_gateway_url() -> None:
    """The doc claims above are backed by real behaviour, not just wording -- construction fails
    closed with no LITELLM_BASE_URL, in every environment (no `environment`/`auth_mode` special
    case)."""
    with pytest.raises(ValidationError, match="LITELLM_BASE_URL"):
        Settings(
            litellm_base_url=None,
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
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


def test_architecture_svg_claims_only_what_the_code_enforces() -> None:
    # The hero diagram (scripts/render_architecture.py) states three guarantees in plain words;
    # each must remain something the code actually does, and it must not resurrect the retired
    # per-member-permissions claim.
    assert "the logged-in user's permissions" not in ARCHITECTURE_SVG
    assert "never hold a database password" in ARCHITECTURE_SVG
    assert "a person approves that exact change" in ARCHITECTURE_SVG
    assert "Row-Level Security forced on every table" in ARCHITECTURE_SVG
    assert "fails closed" in ARCHITECTURE_SVG


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
    # Every host the allow-list object actually contains is named in the doc -- not a hand-copied
    # second list that could silently drift from it.
    allow_list = ResidencyAllowList.load()
    for residency in allow_list.residencies:
        route = allow_list.route_for(residency)
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
    # Spec A4 / #111: the doc names the one object every caller resolves through, not the
    # retired `RESIDENCY_ALLOW_LIST`/`RESIDENCY_MODEL_ALLOW_LIST` globals.
    assert "ResidencyAllowList" in section
    assert "residency_allow_list" in section
    assert "docker/litellm/config.yaml" in section
    assert "trace_sink_host" in section


def test_mcp_server_docs_state_the_client_model_boundary_and_snippet_cap() -> None:
    normalized = " ".join(MCP_SERVER_SOURCE.split())
    assert "outside this application's processor chain" in normalized
    assert "outside residency enforcement" in normalized
    assert "own model" in normalized
    # The existing cap this boundary statement names as the bound that still applies.
    assert "limit = max(1, min(limit, 20))" in MCP_SERVER_SOURCE


def test_residency_doc_no_longer_names_mcp_over_http_as_still_open() -> None:
    # #49 landed (streamable-http with per-connection identity) -- the doc must not still claim
    # it as open, unfinished work the way it did before this ticket (#50).
    section = RESIDENCY_MD.split("## The MCP boundary", 1)[1]
    assert "still-open work of issue #49" not in section
    assert "not a production-ready path" not in section
    assert "stdio" in section
    assert "streamable-http" in section
    assert "production path" in section


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
    assert "ResidencyAllowList" in doc
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


ADR_0006 = (REPO_ROOT / "docs" / "adr" / "0006-server-side-conversations.md").read_text(
    encoding="utf-8"
)


def test_adr_0006_is_accepted_not_proposed() -> None:
    status_line = next(line for line in ADR_0006.splitlines() if line.startswith("- **Status:**"))
    assert "accepted" in status_line
    assert "proposed" not in status_line


def test_adr_0006_states_the_default_retention_period_and_the_job_that_enforces_it() -> None:
    """#35: the ADR now describes the retention period/job as shipped -- with the actual default
    number, not just "a safe default" left unstated -- and names the job that runs it."""
    decision_section = " ".join(
        ADR_0006.split("## Decision", 1)[1].split("## Consequences", 1)[0].split()
    )
    assert "90 days" in decision_section
    assert "app/retention.py" in decision_section
    assert "scripts/retention.py" in decision_section
    assert "tenant_session" in decision_section


def test_claude_md_per_table_rule_names_conversations_retention_with_no_exception() -> None:
    """The project's per-table architecture rule (rule 2) states that conversations/messages are
    governed by the tenant's retention period with no exception -- not only readable from the
    0020 migration."""
    rule_2 = next(line for line in CLAUDE_MD.splitlines() if line.strip().startswith("2. **Every"))
    section_start = CLAUDE_MD.index(rule_2)
    section_end = CLAUDE_MD.index("\n3. ", section_start)
    section = " ".join(CLAUDE_MD[section_start:section_end].split())

    assert "conversations" in section
    assert "messages" in section
    assert "retention" in section
    assert "no exception" in section
    assert "90" in section


def test_claude_md_and_readme_document_the_retention_script_command() -> None:
    assert "scripts/retention.py" in CLAUDE_MD
    assert "scripts/retention.py" in README


def test_readme_table_states_the_default_retention_period() -> None:
    row = next(line for line in README.splitlines() if line.startswith("| Retention |"))
    assert "90 days" in row
    assert "retention_days" in row
    assert "ADR-0006" in row


# --- Spec 6's closing ticket (#50): connecting to and administering the tools server ---

MCP_JSON_EXAMPLE_PATH = REPO_ROOT / ".mcp.json.example"
MCP_CONNECTION_MD = (REPO_ROOT / "docs" / "mcp-connection.md").read_text(encoding="utf-8")
ADR_0005 = (REPO_ROOT / "docs" / "adr" / "0005-agent-identities.md").read_text(encoding="utf-8")


def test_adr_0005_is_accepted_not_proposed() -> None:
    status_line = next(line for line in ADR_0005.splitlines() if line.startswith("- **Status:**"))
    assert "accepted" in status_line
    assert "proposed" not in status_line


def test_mcp_json_example_parses_from_a_different_working_directory() -> None:
    # The point of #50's fix: reading (and, for the stdio entry, running) this file no longer
    # assumes the caller's own working directory happens to be the repo root -- proven by
    # actually reading it from an unrelated cwd, not just by inspecting the string.
    cwd_before = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.chdir(tmp)
            data = json.loads(MCP_JSON_EXAMPLE_PATH.read_text(encoding="utf-8"))
        finally:
            os.chdir(cwd_before)
    assert set(data["mcpServers"]) == {"ai-app-tools", "ai-app-tools-remote"}


def test_mcp_json_example_has_a_valid_local_stdio_shape() -> None:
    data = json.loads(MCP_JSON_EXAMPLE_PATH.read_text(encoding="utf-8"))
    stdio = data["mcpServers"]["ai-app-tools"]
    assert stdio["command"] == "uv"
    # Explicit project directory, not an assumed cwd (the review's "relies on the process cwd
    # for .env" finding, docs/reviews/2026-09-12-security-review.md:305).
    assert "--directory" in stdio["args"]
    assert "MCP_TENANT_ID" in stdio["env"]
    assert "MCP_IDENTITY_ID" in stdio["env"]


def test_mcp_json_example_has_a_valid_networked_shape_with_a_token() -> None:
    data = json.loads(MCP_JSON_EXAMPLE_PATH.read_text(encoding="utf-8"))
    networked = data["mcpServers"]["ai-app-tools-remote"]
    assert networked["type"] == "http"
    assert "/mcp" in networked["url"]
    assert "TENANT_ID" in networked["url"]
    assert "Bearer" in networked["headers"]["Authorization"]


def test_settings_load_from_env_example_file_without_error() -> None:
    # Every new MCP-transport / agent-token setting this spec introduces (MCP_TRANSPORT,
    # JWT_VERIFICATION_KEY, JWT_ALGORITHM, AGENT_TOKEN_SIGNING_KEY, AGENT_TOKEN_ALGORITHM,
    # AGENT_TOKEN_VERIFICATION_KEY, AGENT_TOKEN_TTL_SECONDS) has a working default -- loading the
    # example file constructs cleanly with no override needed for any of them.
    # EMBEDDING_PROVIDER/EMBEDDING_MODEL are deliberately blank in .env.example (ADR-0008: no
    # default, must be set explicitly per deployment) so those two, pre-existing, unrelated
    # required fields are the only ones supplied here.
    settings = Settings(
        _env_file=str(REPO_ROOT / ".env.example"),
        _env_ignore_empty=True,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
    )
    assert settings.mcp_transport == "stdio"
    assert settings.jwt_algorithm == "RS256"
    # Deliberately independent of jwt_algorithm above (review finding, Spec 6): a real IdP signs
    # asymmetrically, this application's own agent tokens are signed/verified symmetrically by
    # default -- see AGENT_TOKEN_ALGORITHM's comment block in .env.example.
    assert settings.agent_token_algorithm == "HS256"
    assert settings.agent_token_ttl_seconds == 300
    assert settings.jwt_verification_key is None
    assert settings.agent_token_signing_key is None
    assert settings.agent_token_verification_key is None
    # Issue #116: blank by default (streamable-http is not the default transport), but the
    # setting name itself must be present in .env.example, not only in code.
    assert settings.mcp_allowed_hosts_list == []


def test_env_example_documents_mcp_allowed_hosts() -> None:
    """Issue #116: `check_mcp_mode` refuses `streamable-http` without this setting -- it must be
    discoverable in `.env.example`, not only in `app/config.py`."""
    assert "MCP_ALLOWED_HOSTS" in ENV_EXAMPLE


def test_connection_guide_documents_mcp_allowed_hosts() -> None:
    """Issue #116: `docs/mcp-connection.md` names the setting that makes the streamable-http
    transport's Host-header allow-list configurable, not only its pre-existing token-verifier
    requirement."""
    assert "MCP_ALLOWED_HOSTS" in MCP_CONNECTION_MD


def test_connection_guide_routes_match_the_agent_identity_and_token_routes() -> None:
    # Cross-referenced one-to-one against the actual routes app/api/agent_identities.py and
    # app/api/agent_tokens.py define, so the guide can't silently drift from the real API surface.
    assert "POST /v1/t/{tenant_id}/agent-identities" in MCP_CONNECTION_MD
    assert "POST /v1/t/{tenant_id}/agent-identities/{identity_id}/credentials" in MCP_CONNECTION_MD
    assert "POST /v1/t/{tenant_id}/agent-tokens" in MCP_CONNECTION_MD
    assert "GET /v1/t/{tenant_id}/agent-credentials" in MCP_CONNECTION_MD
    assert "POST /v1/t/{tenant_id}/agent-credentials/{credential_id}/revoke" in MCP_CONNECTION_MD


def test_connection_guide_covers_both_the_member_and_the_admin_walkthrough() -> None:
    assert "As a member" in MCP_CONNECTION_MD
    assert "As a tenant admin" in MCP_CONNECTION_MD
    assert "stdio" in MCP_CONNECTION_MD
    assert "streamable-http" in MCP_CONNECTION_MD


def test_claude_md_directory_table_marks_networked_transport_as_production() -> None:
    line = next(line for line in CLAUDE_MD.splitlines() if line.startswith("app/mcp/server.py"))
    assert "streamable-http" in line
    assert "production" in line
    assert "stdio" in line
    assert "development" in line


def test_claude_md_commands_mark_the_stdio_launch_command_as_development_only() -> None:
    line = next(
        line for line in CLAUDE_MD.splitlines() if "uv run python -m app.mcp.server" in line
    )
    assert "development" in line.lower()
    assert "streamable-http" in line


def test_readme_and_claude_md_state_a_token_is_required_outside_local_development() -> None:
    mcp_row = next(line for line in README.splitlines() if line.startswith("| MCP server |"))
    assert "token" in mcp_row.lower()
    assert "production" in mcp_row.lower()

    make_it_your_own_section = README.split("## Make it your own", 1)[1]
    assert "mcp-connection.md" in make_it_your_own_section
    assert "token is required outside local development" in make_it_your_own_section


ADR_0007 = (REPO_ROOT / "docs" / "adr" / "0007-approval-of-writing-tools.md").read_text(
    encoding="utf-8"
)


def test_adr_0007_is_accepted_not_proposed() -> None:
    # Spec 5's closing ticket (#42): the whole approval mechanism (pending actions, standing
    # grants, the two-agent split, run limits) is now real, so the ADR moves from `proposed` to
    # `accepted` per this repo's own convention (docs/agents/domain.md).
    status_line = next(line for line in ADR_0007.splitlines() if line.startswith("- **Status:**"))
    assert "accepted" in status_line
    assert "proposed" not in status_line


def test_adr_0007_states_these_tables_are_ordinary_tenant_tables_deferring_erasure_to_spec_9() -> (
    None
):
    assert "ordinary tenant tables" in ADR_0007
    assert "Spec 9" in ADR_0007
    assert "removed with it" in ADR_0007 or "removed with the tenant" in ADR_0007


def test_claude_md_writing_tool_rule_names_pending_actions_role_recheck_and_standing_grants() -> (
    None
):
    # #42's content-check acceptance criterion: rule 4 must name all three mechanisms by name,
    # not just gesture at "a confirmation step".
    rule_4 = CLAUDE_MD.split("4. **Agents access data only through tools**", 1)[1].split(
        "\n5. ", 1
    )[0]
    assert "pending action" in rule_4
    assert "standing grant" in rule_4
    assert "re-checks the acting membership's role" in rule_4 or "re-check" in rule_4


def test_claude_md_writing_tool_rule_prohibits_always_allow_for_a_person() -> None:
    rule_4 = CLAUDE_MD.split("4. **Agents access data only through tools**", 1)[1].split(
        "\n5. ", 1
    )[0]
    assert "No derived project may ever add" in rule_4
    assert "always allow" in rule_4.lower()


def test_readme_module_table_names_the_approval_and_two_agent_modules() -> None:
    approvals_row = next(line for line in README.splitlines() if line.startswith("| Approvals |"))
    assert "pending_actions" in approvals_row
    assert "standing_grants" in approvals_row
    assert "approval_audit" in approvals_row

    agents_row = next(line for line in README.splitlines() if line.startswith("| Agents |"))
    assert "one-shot agent" in agents_row
    assert "chat agent" in agents_row


def test_readme_four_rules_summary_names_the_approval_mechanism() -> None:
    rules_section = " ".join(
        README.split("## The four rules that hold it together", 1)[1].split("## ", 1)[0].split()
    )
    assert "writing tool requires approval" in rules_section
    assert "standing grant" in rules_section
    assert "always allow" in rules_section.lower()
