"""The schema for `public.tenants.settings` (Spec 10 / #78, ADR-0002, ADR-0011).

Also the home of the tenant-editable retention period (ADR-0006, Spec 4 / #35): how long a
tenant's conversations and messages live, measured from `conversations.last_activity_at`, before
the retention job (`app/retention.py`, `scripts/retention.py`) deletes them. A tenant's admin sets
`settings["retention_days"]` like any other tenant-editable setting, read back out through
`app.repositories.tenant_settings.TenantSettingsRepository` -- the one place any caller resolves
this JSONB column, exactly like `app.observability` already does for `content_tracing_opt_in`.
`DEFAULT_RETENTION_DAYS` below is the documented, conservative fallback used for every tenant that
never sets one.

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

## Catalog (ADR-0008, ADR-0011): every setting the control-plane schema currently defines

The catalog of `public.tenants.settings`/`control.tenants` fields a request may read or write.
A new field (tenant-editable or operator-owned) is documented here, in this one place, rather
than left to whichever module happens to read it first.

- **`residency`** -- home: `control.tenants.residency` (operator-owned, migration
  `0013_control_plane_residency.py`). Shape: `text`, nullable. Validation: a `CHECK` constraint
  restricting it to one of `Settings.residency_allow_list.residencies`
  (`app.residency.ResidencyAllowList`); a tenant's own request can never write it (no `UPDATE`
  grant on `control.tenants`). Default: **no default** -- a tenant without one fails closed
  (`app.residency.ResidencyUnresolved`), never inherits another jurisdiction's route.
- **`model`** -- home: `public.tenants.settings["model"]` (tenant-editable, this module). Shape:
  `str` or `None`. Validation: validated against `Settings.residency_allow_list.alias_for` for
  the tenant's own residency, at the point a chat model is resolved
  (`app.llm.resolve_tenant_chat_model`/`validate_model_for_residency`), not by this model itself.
  Default: `None` (the deployment default `Settings.llm_model` applies).
- **`content_tracing_opt_in`** -- home: `public.tenants.settings["content_tracing_opt_in"]`
  (tenant-editable, this module). Shape: `bool`. Validation: plain Pydantic bool coercion, no
  allow-list -- any tenant admin may flip its own tenant's flag. Default: `False` -- content-free
  tracing (`include_content=False`) until explicitly turned on, see `app.observability`.

`residency` is deliberately never a field on `TenantSettings` below (see the class docstring): it
lives in `control.tenants`, read through `app.repositories.control.ControlRepository`, not
through `TenantSettingsRepository`. It is listed in this catalog anyway because it is resolved
alongside `content_tracing_opt_in` at the exact same call sites
(`app.observability.resolve_tenant_tracing_selection`, `app.residency.resolve_residency_route`)
and a reader of one needs to see the other.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator

# The conservative default retention period (days) for a tenant's conversations and messages
# (ADR-0006) when the tenant has never set `settings["retention_days"]` itself: long enough to
# support a review of a recent exchange or an approval dispute, short enough that the pooled
# tenants table does not grow into an unbounded, indefinite record of everyone's conversations.
# Stated here (not only readable from a migration) so it is the one place both the job and the
# project's own docs cite.
DEFAULT_RETENTION_DAYS = 90

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

    # `frozen`: a validated settings object rides on the immutable `app.tenant_record.TenantRecord`
    # for a whole request (#104) -- nothing may change it half-way through.
    model_config = ConfigDict(extra="forbid", frozen=True)

    # `tenants.settings["model"]` (app/llm.py): per-tenant model override. Residency is not
    # here on purpose: it is an operator-owned control-plane fact (`control.tenants.residency`,
    # ADR-0008), so a tenant cannot move itself to another jurisdiction.
    model: str | None = None

    # `tenants.settings["retention_days"]` (ADR-0006, `app/retention.py`): how many days of no
    # activity (`conversations.last_activity_at`) a conversation may go before the retention job
    # deletes it and its messages. `None` (never set) means "use DEFAULT_RETENTION_DAYS" -- a
    # tenant can only ever make its own retention *shorter or longer*, never disable the job.
    retention_days: PositiveInt | None = None

    # `tenants.settings["content_tracing_opt_in"]` (Spec 8 / #62, ADR-0008): whether this
    # tenant's agent runs may include prompts, tool arguments, and document text in the spans
    # sent to the trace sink. Off by default -- tracing always carries identifiers, timings, and
    # errors, never content, until the tenant admin explicitly opts in. Deliberately a tenant
    # preference (not an operator-owned control-plane fact like residency): ADR-0008's user
    # stories describe it as something "a tenant admin" turns on for their own tenant, scoped to
    # the whole tenant, not a per-run toggle -- see `app.observability`, which is the one place
    # this value is turned into an actual `InstrumentationSettings.include_content`.
    content_tracing_opt_in: bool = False

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
