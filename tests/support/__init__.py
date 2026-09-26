"""Test support package: one embedded Postgres cluster, one seeding function, shared across the
integration suite (issue #96 / spec #90, "A6").

Importable only by files that opt into the embedded-Postgres dependency: importing this package
imports `pgserver`, guarded right here by `pytest.importorskip`, so the no-database slice of the
suite (most of it) never needs `pgserver` installed and collecting it is unaffected by this
package existing.

Before this package, seventeen integration test files each hand-wrote their own embedded-cluster
boot, role bootstrap, `psql` runner, and tenant/membership/document seeding. This package holds
exactly one copy of each: `cluster`/`environment` (`tests.support.cluster`) for the former,
`seed_tenant`/`seed_membership` (`tests.support.seeding`) for the latter.
"""

from __future__ import annotations

import pytest

pgserver = pytest.importorskip("pgserver")

from tests.support.cluster import ROLE_BOOTSTRAP_SQL, Cluster, cluster, environment  # noqa: E402
from tests.support.seeding import (  # noqa: E402
    SeededTenant,
    UnknownNotNullColumnError,
    assert_known_not_null_columns,
    seed_membership,
    seed_tenant,
    vector_literal,
)

__all__ = [
    "ROLE_BOOTSTRAP_SQL",
    "Cluster",
    "SeededTenant",
    "UnknownNotNullColumnError",
    "assert_known_not_null_columns",
    "cluster",
    "environment",
    "seed_membership",
    "seed_tenant",
    "vector_literal",
]
