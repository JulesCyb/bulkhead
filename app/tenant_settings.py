"""The schema for `public.tenants.settings` (Spec 10 / #78, ADR-0002, ADR-0011).

`tenants.settings` is a free-form JSONB column the `app` role may write on the tenant's own row
(the one column-level write grant a tenant's own request has, per #12/ADR-0011 — every other
column of `control.tenants`/`public.tenants` is read-only to a tenant). Isolation tier and
database alias are operator-owned facts that live in `control.tenants` instead (#73) and are
never accepted here, but nothing stops a future settings-update endpoint from being built with a
raw `dict[str, Any]` that just happens to also accept a key shaped like one of those facts, or a
value that is itself a connection string. `TenantSettings` is the one seam any code that writes
`tenants.settings` is meant to validate through, so that mistake is rejected at the schema level
instead of trusted to every call site's own judgement.

Two things are rejected, independent of whether a field name is otherwise known to this model:

- a key shaped like a database alias, isolation tier, or DSN/credential field (`database_alias`,
  `db_alias`, `isolation_tier`, `dsn`, `database_url`, `connection_string`, ...) — see
  `_FORBIDDEN_KEY_MARKERS` below;
- a string *value* that looks like a connection string (`scheme://...`) for any key at all, since
  a tenant could otherwise smuggle one into an allowed field such as `model`.

`extra="forbid"` catches every other unknown key on top of that — a tenant may only ever set the
fields this model actually declares.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

# Substrings that mark a settings key as naming a database alias, isolation tier, or credential/
# connection-string field rather than a genuine tenant preference — checked against the
# normalized (lowercased, `-` -> `_`) key, so `database-alias`, `DatabaseAlias`, and
# `database_alias` are all caught the same way.
_FORBIDDEN_KEY_MARKERS: tuple[str, ...] = (
    "database_alias",
    "db_alias",
    "isolation_tier",
    "dsn",
    "database_url",
    "db_url",
    "connection_string",
    "connection_str",
    "credential",
    "password",
    "secret",
)

# A value shaped like `scheme://...` — the general form of a DSN/connection string
# (postgres://, postgresql://, postgresql+asyncpg://, mysql://, ...). Rejected in *any* field,
# not only ones with a suspicious name.
_DSN_LIKE_VALUE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


class TenantEditableSettingRejected(ValueError):
    """Raised (wrapped in a pydantic ValidationError) when a `tenants.settings` key or value
    names or carries a database alias, isolation tier, or connection string/credential — facts
    that ADR-0002/ADR-0011 keep exclusively in the operator-owned control plane."""


class TenantSettings(BaseModel):
    """The tenant-editable subset of `tenants.settings`. Unknown keys are rejected outright
    (`extra="forbid"`); the checks below reject alias/isolation-tier/DSN-shaped input even for
    keys this model does not itself declare, so a future field added here can never accidentally
    reopen that hole.
    """

    model_config = ConfigDict(extra="forbid")

    # `tenants.settings["model"]` (app/llm.py): per-tenant model override. Residency is not
    # here on purpose: it is an operator-owned control-plane fact (`control.tenants.residency`,
    # ADR-0008), so a tenant cannot move itself to another jurisdiction.
    model: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_alias_isolation_and_dsn_shaped_input(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for key, value in data.items():
            normalized = _normalize_key(str(key))
            if any(marker in normalized for marker in _FORBIDDEN_KEY_MARKERS):
                raise TenantEditableSettingRejected(
                    f"{key!r} is not a tenant-editable setting: database aliases, isolation "
                    "tiers, and connection strings/credentials are operator-owned facts in the "
                    "control plane (ADR-0002, ADR-0011), never a tenant's own setting."
                )
            if isinstance(value, str) and _DSN_LIKE_VALUE.match(value.strip()):
                raise TenantEditableSettingRejected(
                    f"{key!r} looks like a connection string ({value!r}); a tenant setting may "
                    "never hold a DSN, hostname, or credential."
                )
        return data
