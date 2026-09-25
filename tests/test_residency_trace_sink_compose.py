"""Rendered `docker-compose.yml` configuration (Spec 8 / #64): the trace sink is a
content-bearing, residency-governed endpoint exactly like the database and gateway hosts
(ADR-0008) — it must reach the two services that need it (`api`, which runs agent traffic, and
`migrate`, which runs against the same residency-governed deployment) and must never reach a
service that has no business receiving it (`postgres`, `litellm`).

`api` and `migrate` each declare an explicit `environment:` block in docker-compose.yml (Spec 1 /
#18 replaced their blanket `env_file: .env` with the exact variables each one needs) that names
`LANGFUSE_HOST`/`RESIDENCY` among them; `postgres` and `litellm` only ever get the specific
variables named under their own `environment:` block. This test renders the real compose file
and asserts that wiring directly, so an edit that stops naming the trace sink for a service that
needs it — or that starts forwarding it to a service that shouldn't have it — fails here instead
of in an incident.

Rendering itself (Docker availability check, throwaway `.env`, `docker compose config`) is
shared with the other compose tests via tests/compose_helpers.py (Spec 1 / #18).
"""

from __future__ import annotations

import pytest

from tests.compose_helpers import docker_compose_available, render_compose, temporary_env_file

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

pytestmark = pytest.mark.skipif(
    not docker_compose_available(), reason="docker compose CLI/daemon not available"
)


@pytest.fixture
def env_file():
    with temporary_env_file(
        {"LANGFUSE_HOST": TEST_LANGFUSE_HOST, "RESIDENCY": TEST_RESIDENCY}
    ) as path:
        yield path


@pytest.mark.parametrize("service_name", SERVICES_THAT_NEED_THE_TRACE_SINK)
def test_trace_sink_and_residency_reach_services_that_need_them(env_file, service_name):
    rendered = render_compose()
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
    rendered = render_compose()
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
    rendered = render_compose()
    accounted_for = set(SERVICES_THAT_NEED_THE_TRACE_SINK) | set(
        SERVICES_WITH_NO_BUSINESS_RECEIVING_IT
    )
    assert set(rendered["services"]) == accounted_for
