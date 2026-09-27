"""The residency allow-list is data, loaded and validated by one object (ADR-0008, spec A4 /
#94, #110): `app.residency.ResidencyAllowList.load` reads `config/residency.toml` (or an explicit
path) with stdlib `tomllib`; `from_data` runs the exact same validation over an in-memory dict, the
seam every test below builds against instead of a file on disk. This file also holds the property
that closes the original finding (#63): two residencies must never be able to reach the same host,
or the allow-list's jurisdictional promise is fiction.
"""

from __future__ import annotations

import textwrap

import pytest

from app.residency import ResidencyAllowList, ResidencyConfigError, ResidencyUnresolved


def _write(tmp_path, contents: str):
    path = tmp_path / "residency.toml"
    path.write_text(textwrap.dedent(contents), encoding="utf-8")
    return path


# --- Loading a well-formed file --------------------------------------------------------------


def test_loads_the_shipped_config_file_with_at_least_two_disjoint_residencies():
    allow_list = ResidencyAllowList.load()
    assert set(allow_list.residencies) >= {"eu", "us"}
    for residency in allow_list.residencies:
        route = allow_list.route_for(residency)
        assert route.model_host_patterns
        assert route.embedding_endpoint
        assert route.trace_sink_host


def test_loads_a_well_formed_file_from_an_explicit_path(tmp_path):
    path = _write(
        tmp_path,
        """
        [residency.acme]
        model_host_patterns = ["gateway-acme.internal"]
        embedding_endpoint = "https://gateway-acme.internal/v1"
        trace_sink_host = "acme.cloud.langfuse.com"
        models = ["claude-acme"]
        """,
    )
    allow_list = ResidencyAllowList.load(path)
    assert allow_list.residencies == ("acme",)
    assert allow_list.route_for("acme").model_host_patterns == ("gateway-acme.internal",)
    assert allow_list.model_aliases("acme") == ("claude-acme",)


def test_models_key_is_optional():
    """A residency with no gateway model aliases configured is a residency with no entry in the
    model allow-list -- never an empty tuple standing in for "anything goes"."""
    allow_list = ResidencyAllowList.from_data(
        {
            "residency": {
                "acme": {
                    "model_host_patterns": ["gateway-acme.internal"],
                    "embedding_endpoint": "https://gateway-acme.internal/v1",
                    "trace_sink_host": "acme.cloud.langfuse.com",
                }
            }
        }
    )
    assert "acme" in allow_list.residencies
    assert allow_list.model_aliases("acme") == ()


# --- from_data: the in-memory seam every test above and below uses ---------------------------


def test_from_data_builds_the_same_shape_load_does():
    allow_list = ResidencyAllowList.from_data(
        {
            "residency": {
                "acme": {
                    "model_host_patterns": ["gateway-acme.internal"],
                    "embedding_endpoint": "https://gateway-acme.internal/v1",
                    "trace_sink_host": "acme.cloud.langfuse.com",
                    "models": ["claude-acme"],
                }
            }
        }
    )
    assert allow_list.residencies == ("acme",)
    assert allow_list.route_for("acme").embedding_endpoint == "https://gateway-acme.internal/v1"
    assert allow_list.model_aliases("acme") == ("claude-acme",)


# --- Fail-closed on a broken file/dict ---------------------------------------------------------


def test_missing_file_fails_closed_with_a_clear_message(tmp_path):
    missing = tmp_path / "does-not-exist.toml"
    with pytest.raises(ResidencyConfigError, match="not found or unreadable"):
        ResidencyAllowList.load(missing)


def test_malformed_toml_fails_closed_with_a_clear_message(tmp_path):
    path = _write(tmp_path, "this is not [valid toml")
    with pytest.raises(ResidencyConfigError, match="not valid TOML"):
        ResidencyAllowList.load(path)


def test_empty_data_fails_closed():
    with pytest.raises(ResidencyConfigError, match=r"\[residency\.<name>\]"):
        ResidencyAllowList.from_data({})


def test_residency_entry_missing_a_required_key_fails_closed():
    with pytest.raises(ResidencyConfigError, match="missing required key"):
        ResidencyAllowList.from_data(
            {
                "residency": {
                    "acme": {
                        "model_host_patterns": ["gateway-acme.internal"],
                        "trace_sink_host": "acme.cloud.langfuse.com",
                    }
                }
            }
        )


def test_empty_model_host_patterns_fails_closed():
    with pytest.raises(ResidencyConfigError, match="model_host_patterns"):
        ResidencyAllowList.from_data(
            {
                "residency": {
                    "acme": {
                        "model_host_patterns": [],
                        "embedding_endpoint": "https://gateway-acme.internal/v1",
                        "trace_sink_host": "acme.cloud.langfuse.com",
                    }
                }
            }
        )


def test_non_table_residency_entry_fails_closed():
    with pytest.raises(ResidencyConfigError):
        ResidencyAllowList.from_data({"residency": "not a table of tables"})


# --- The property this fix exists for: residencies never share a host --------------------------


def test_two_residencies_sharing_a_model_host_pattern_fails_closed():
    with pytest.raises(ResidencyConfigError, match="must never share a host"):
        ResidencyAllowList.from_data(
            {
                "residency": {
                    "eu": {
                        "model_host_patterns": ["*.anthropic.com"],
                        "embedding_endpoint": "https://gateway-eu.internal/v1",
                        "trace_sink_host": "eu.cloud.langfuse.com",
                    },
                    "us": {
                        "model_host_patterns": ["*.anthropic.com"],
                        "embedding_endpoint": "https://gateway-us.internal/v1",
                        "trace_sink_host": "us.cloud.langfuse.com",
                    },
                }
            }
        )


def test_two_residencies_sharing_an_embedding_host_fails_closed():
    with pytest.raises(ResidencyConfigError, match="must never share a host"):
        ResidencyAllowList.from_data(
            {
                "residency": {
                    "eu": {
                        "model_host_patterns": ["gateway-eu.internal"],
                        "embedding_endpoint": "https://api.openai.com/v1",
                        "trace_sink_host": "eu.cloud.langfuse.com",
                    },
                    "us": {
                        "model_host_patterns": ["gateway-us.internal"],
                        "embedding_endpoint": "https://api.openai.com/v1",
                        "trace_sink_host": "us.cloud.langfuse.com",
                    },
                }
            }
        )


def test_two_residencies_sharing_a_trace_sink_fails_closed():
    with pytest.raises(ResidencyConfigError, match="must never share a host"):
        ResidencyAllowList.from_data(
            {
                "residency": {
                    "eu": {
                        "model_host_patterns": ["gateway-eu.internal"],
                        "embedding_endpoint": "https://gateway-eu.internal/v1",
                        "trace_sink_host": "shared.cloud.langfuse.com",
                    },
                    "us": {
                        "model_host_patterns": ["gateway-us.internal"],
                        "embedding_endpoint": "https://gateway-us.internal/v1",
                        "trace_sink_host": "shared.cloud.langfuse.com",
                    },
                }
            }
        )


def test_the_shipped_allow_list_shares_no_host_across_residencies():
    """Property test over the loaded (shipped) allow-list, not just from_data's own validation of
    a hand-crafted fixture: every host/pattern named for one residency is absent from every other
    residency's own set."""
    allow_list = ResidencyAllowList.load()
    per_residency_hosts: dict[str, set[str]] = {}
    for residency in allow_list.residencies:
        route = allow_list.route_for(residency)
        embedding_host = route.embedding_endpoint.split("//", 1)[-1].split("/", 1)[0]
        per_residency_hosts[residency] = {
            *route.model_host_patterns,
            embedding_host,
            route.trace_sink_host,
        }

    residencies = list(per_residency_hosts)
    for i, a in enumerate(residencies):
        for b in residencies[i + 1 :]:
            overlap = per_residency_hosts[a] & per_residency_hosts[b]
            assert not overlap, f"{a!r} and {b!r} both allow-list {overlap}"


def test_no_residency_allow_lists_a_global_provider_domain():
    """The original finding (#63): a global, every-jurisdiction-reachable provider domain
    (*.anthropic.com, *.openai.com) enforces nothing if it sits on every residency's allow-list.
    Every model_host_patterns entry must be gateway-specific, never one of these."""
    banned = {"*.anthropic.com", "*.openai.com", "api.anthropic.com", "api.openai.com"}
    allow_list = ResidencyAllowList.load()
    for residency in allow_list.residencies:
        route = allow_list.route_for(residency)
        assert not (set(route.model_host_patterns) & banned), (
            f"residency {residency!r} allow-lists a global provider domain: "
            f"{route.model_host_patterns}"
        )
        embedding_host = route.embedding_endpoint.split("//", 1)[-1].split("/", 1)[0]
        assert embedding_host not in banned, (
            f"residency {residency!r} embedding_endpoint {route.embedding_endpoint!r} is a "
            "global provider domain"
        )


# --- route_for / alias_for / model_aliases: the lookups (#94's "one function to call") --------


def _acme_us_allow_list() -> ResidencyAllowList:
    return ResidencyAllowList.from_data(
        {
            "residency": {
                "eu": {
                    "model_host_patterns": ["gateway-eu.internal"],
                    "embedding_endpoint": "https://gateway-eu.internal/v1",
                    "trace_sink_host": "eu.cloud.langfuse.com",
                    "models": ["claude-eu", "embeddings"],
                },
                "us": {
                    "model_host_patterns": ["gateway-us.internal"],
                    "embedding_endpoint": "https://gateway-us.internal/v1",
                    "trace_sink_host": "us.cloud.langfuse.com",
                    "models": ["claude"],
                },
            }
        }
    )


def test_route_for_unknown_residency_raises_the_one_exception_type():
    allow_list = _acme_us_allow_list()
    with pytest.raises(ResidencyUnresolved, match="mars"):
        allow_list.route_for("mars")


def test_alias_for_strips_a_provider_prefix():
    allow_list = _acme_us_allow_list()
    assert allow_list.alias_for("eu", "anthropic:claude-eu") == "claude-eu"
    assert allow_list.alias_for("eu", "claude-eu") == "claude-eu"


def test_alias_for_unlisted_model_raises_the_one_exception_type():
    allow_list = _acme_us_allow_list()
    with pytest.raises(ResidencyUnresolved, match="mistral-large"):
        allow_list.alias_for("eu", "mistral-large")


def test_alias_for_is_residency_specific():
    """`claude` is on the US list but not the EU one -- proves the check is against the given
    residency, not any residency."""
    allow_list = _acme_us_allow_list()
    assert allow_list.alias_for("us", "claude") == "claude"
    with pytest.raises(ResidencyUnresolved):
        allow_list.alias_for("eu", "claude")


def test_model_aliases_of_a_residency_with_none_configured_is_empty():
    allow_list = ResidencyAllowList.from_data(
        {
            "residency": {
                "acme": {
                    "model_host_patterns": ["gateway-acme.internal"],
                    "embedding_endpoint": "https://gateway-acme.internal/v1",
                    "trace_sink_host": "acme.cloud.langfuse.com",
                }
            }
        }
    )
    assert allow_list.model_aliases("acme") == ()


def test_residency_config_error_is_a_residency_unresolved():
    """One exception type for every fail-closed lookup/validation in this module (module
    docstring): a bad config file and a bad lookup against a good one are the same class of
    failure, so a caller that only wants to fail closed can catch the one base type."""
    assert issubclass(ResidencyConfigError, ResidencyUnresolved)
