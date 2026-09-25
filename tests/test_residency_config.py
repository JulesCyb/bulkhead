"""The residency allow-list is data loaded from a file, not a Python literal (ADR-0008, Spec 8 /
#63): `app.config.load_residency_config` reads `config/residency.toml` (or `RESIDENCY_CONFIG_PATH`)
with stdlib `tomllib`, validates it, and fails closed -- naming the file and the exact problem --
on anything malformed, empty, or missing. This file also holds the property that closes the
original finding: two residencies must never be able to reach the same host, or the allow-list's
jurisdictional promise is fiction.
"""

from __future__ import annotations

import textwrap

import pytest

from app.config import RESIDENCY_ALLOW_LIST, ResidencyConfigError, load_residency_config


def _write(tmp_path, contents: str):
    path = tmp_path / "residency.toml"
    path.write_text(textwrap.dedent(contents), encoding="utf-8")
    return path


# --- Loading a well-formed file ------------------------------------------------------------


def test_loads_the_shipped_config_file_with_at_least_two_disjoint_residencies():
    allow_list, model_allow_list = load_residency_config()
    assert set(allow_list) >= {"eu", "us"}
    assert set(model_allow_list) >= {"eu", "us"}
    for route in allow_list.values():
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
    allow_list, model_allow_list = load_residency_config(path)
    assert set(allow_list) == {"acme"}
    assert allow_list["acme"].model_host_patterns == ("gateway-acme.internal",)
    assert model_allow_list["acme"] == ("claude-acme",)


def test_models_key_is_optional():
    """A residency with no gateway model aliases configured is a residency with no entry in the
    model allow-list -- never an empty tuple standing in for "anything goes" (mirrors the
    reasoning already documented for RESIDENCY_MODEL_ALLOW_LIST)."""

    def _write_and_load(tmp_path):
        path = _write(
            tmp_path,
            """
            [residency.acme]
            model_host_patterns = ["gateway-acme.internal"]
            embedding_endpoint = "https://gateway-acme.internal/v1"
            trace_sink_host = "acme.cloud.langfuse.com"
            """,
        )
        return load_residency_config(path)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        allow_list, model_allow_list = _write_and_load(Path(d))
        assert "acme" in allow_list
        assert "acme" not in model_allow_list


# --- Fail-closed on a broken file ----------------------------------------------------------


def test_missing_file_fails_closed_with_a_clear_message(tmp_path):
    missing = tmp_path / "does-not-exist.toml"
    with pytest.raises(ResidencyConfigError, match="not found or unreadable"):
        load_residency_config(missing)


def test_malformed_toml_fails_closed_with_a_clear_message(tmp_path):
    path = _write(tmp_path, "this is not [valid toml")
    with pytest.raises(ResidencyConfigError, match="not valid TOML"):
        load_residency_config(path)


def test_empty_file_fails_closed():
    def _write_and_load(tmp_path):
        path = _write(tmp_path, "")
        return load_residency_config(path)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ResidencyConfigError, match=r"\[residency\.<name>\]"):
            _write_and_load(Path(d))


def test_residency_entry_missing_a_required_key_fails_closed(tmp_path):
    path = _write(
        tmp_path,
        """
        [residency.acme]
        model_host_patterns = ["gateway-acme.internal"]
        trace_sink_host = "acme.cloud.langfuse.com"
        """,
    )
    with pytest.raises(ResidencyConfigError, match="missing required key"):
        load_residency_config(path)


def test_empty_model_host_patterns_fails_closed(tmp_path):
    path = _write(
        tmp_path,
        """
        [residency.acme]
        model_host_patterns = []
        embedding_endpoint = "https://gateway-acme.internal/v1"
        trace_sink_host = "acme.cloud.langfuse.com"
        """,
    )
    with pytest.raises(ResidencyConfigError, match="model_host_patterns"):
        load_residency_config(path)


def test_non_table_residency_entry_fails_closed(tmp_path):
    path = _write(tmp_path, 'residency = "not a table of tables"')
    with pytest.raises(ResidencyConfigError):
        load_residency_config(path)


# --- The property this fix exists for: residencies never share a host ----------------------


def test_two_residencies_sharing_a_model_host_pattern_fails_closed(tmp_path):
    path = _write(
        tmp_path,
        """
        [residency.eu]
        model_host_patterns = ["*.anthropic.com"]
        embedding_endpoint = "https://gateway-eu.internal/v1"
        trace_sink_host = "eu.cloud.langfuse.com"

        [residency.us]
        model_host_patterns = ["*.anthropic.com"]
        embedding_endpoint = "https://gateway-us.internal/v1"
        trace_sink_host = "us.cloud.langfuse.com"
        """,
    )
    with pytest.raises(ResidencyConfigError, match="must never share a host"):
        load_residency_config(path)


def test_two_residencies_sharing_an_embedding_host_fails_closed(tmp_path):
    path = _write(
        tmp_path,
        """
        [residency.eu]
        model_host_patterns = ["gateway-eu.internal"]
        embedding_endpoint = "https://api.openai.com/v1"
        trace_sink_host = "eu.cloud.langfuse.com"

        [residency.us]
        model_host_patterns = ["gateway-us.internal"]
        embedding_endpoint = "https://api.openai.com/v1"
        trace_sink_host = "us.cloud.langfuse.com"
        """,
    )
    with pytest.raises(ResidencyConfigError, match="must never share a host"):
        load_residency_config(path)


def test_two_residencies_sharing_a_trace_sink_fails_closed(tmp_path):
    path = _write(
        tmp_path,
        """
        [residency.eu]
        model_host_patterns = ["gateway-eu.internal"]
        embedding_endpoint = "https://gateway-eu.internal/v1"
        trace_sink_host = "shared.cloud.langfuse.com"

        [residency.us]
        model_host_patterns = ["gateway-us.internal"]
        embedding_endpoint = "https://gateway-us.internal/v1"
        trace_sink_host = "shared.cloud.langfuse.com"
        """,
    )
    with pytest.raises(ResidencyConfigError, match="must never share a host"):
        load_residency_config(path)


def test_the_shipped_allow_list_shares_no_host_across_residencies():
    """Property test over the loaded (shipped) allow-list, not just the loader's own validation
    of a hand-crafted fixture: every host/pattern named for one residency is absent from every
    other residency's own set."""
    per_residency_hosts: dict[str, set[str]] = {}
    for residency, route in RESIDENCY_ALLOW_LIST.items():
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
    for residency, route in RESIDENCY_ALLOW_LIST.items():
        assert not (set(route.model_host_patterns) & banned), (
            f"residency {residency!r} allow-lists a global provider domain: "
            f"{route.model_host_patterns}"
        )
        embedding_host = route.embedding_endpoint.split("//", 1)[-1].split("/", 1)[0]
        assert embedding_host not in banned, (
            f"residency {residency!r} embedding_endpoint {route.embedding_endpoint!r} is a "
            "global provider domain"
        )
