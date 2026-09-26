"""Test support package: one embedded Postgres cluster, one seeding function, one gateway admin
fake, shared across the integration suite (issue #96 / spec #90, "A6"; seeding through the
operator commands and the shared gateway fake is issue #98 / "A6-T3").

Importable only by files that opt into the embedded-Postgres dependency: importing this package
imports `pgserver`, guarded right here by `pytest.importorskip`, so the no-database slice of the
suite (most of it) never needs `pgserver` installed and collecting it is unaffected by this
package existing.

Before this package, seventeen integration test files each hand-wrote their own embedded-cluster
boot, role bootstrap, `psql` runner, tenant/membership/document seeding, and (four of them) their
own `httpx.MockTransport`-backed gateway admin fake. This package holds exactly one copy of each:
`cluster`/`environment` (`tests.support.cluster`) for the first, `seed_tenant`/`seed_membership`
(`tests.support.seeding`) for the second, `fake_gateway_admin_client` (`tests.support.gateway`)
for the third.
"""

from __future__ import annotations

import pytest

pgserver = pytest.importorskip("pgserver")

from tests.support.cluster import (  # noqa: E402
    ROLE_BOOTSTRAP_SQL,
    Cluster,
    cluster,
    create_database,
    environment,
    migration_run_without_disrupting_logging,
)
from tests.support.gateway import (  # noqa: E402
    FakeGatewayAdmin,
    fake_gateway_admin_client,
)
from tests.support.seeding import (  # noqa: E402
    SeededTenant,
    UnknownNotNullColumnError,
    assert_known_not_null_columns,
    seed_conversation,
    seed_document,
    seed_membership,
    seed_tenant,
    set_tenant_identity_issuer,
    set_tenant_retention_days,
    vector_literal,
)

__all__ = [
    "ROLE_BOOTSTRAP_SQL",
    "Cluster",
    "FakeGatewayAdmin",
    "SeededTenant",
    "UnknownNotNullColumnError",
    "assert_known_not_null_columns",
    "cluster",
    "create_database",
    "environment",
    "fake_gateway_admin_client",
    "migration_run_without_disrupting_logging",
    "seed_conversation",
    "seed_document",
    "seed_membership",
    "seed_tenant",
    "set_tenant_identity_issuer",
    "set_tenant_retention_days",
    "vector_literal",
]
