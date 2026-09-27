"""The schema for `public.tenants.settings` (Spec 10 / #78, ADR-0002, ADR-0011).

Also the home of the tenant-editable retention period (ADR-0006, Spec 4 / #35): how long a
tenant's conversations and messages live, measured from `conversations.last_activity_at`, before
the retention job (`app/retention.py`, `scripts/retention.py`) deletes them. A tenant's admin sets
`settings["retention_days"]` like any other tenant-editable setting, read back out through
`app.repositories.tenant_settings.TenantSettingsRepository` -- the one place this JSONB column is
resolved, as part of the tenant record (`app.tenant_record.TenantRecord.settings`, #104) that
retention, tracing (`content_tracing_opt_in`), and model resolution (`model`) all read (#105).
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
  `str` or `None`. Validation: against `Settings.residency_allow_list.alias_for` for the
  tenant's own residency, twice, never by this model itself: on write (every writer of
  `tenants.settings` -- today only the operator tool's `create`,
  `app.operator.create._validate_model` -- rejects an unlisted name before writing it) and again
  on every read (`app.llm.resolve_tenant_chat_model`/`validate_model_for_residency`, from the
  tenant record, #105), failing closed exactly like an unlisted deployment default -- so an
  allow-list change after the write can never silently degrade to another model.
  Default: `None` (the deployment default `Settings.llm_model` applies).
- **`content_tracing_opt_in`** -- home: `public.tenants.settings["content_tracing_opt_in"]`
  (tenant-editable, this module). Shape: `bool`. Validation: plain Pydantic bool coercion, no
  allow-list -- any tenant admin may flip its own tenant's flag. Default: `False` -- content-free
  tracing (`include_content=False`) until explicitly turned on, see `app.observability`.
- **`retention_days`** -- home: `public.tenants.settings["retention_days"]` (tenant-editable,
  this module). Shape: `PositiveInt | None`. Validation: construction checks "a positive
  integer"; the deployment cap (#84, ADR-0006; GDPR Art. 5(1)(e)) is enforced twice, exactly like
  `model` above: on write, by this model's own `require_retention_within_cap(max_days=...)`,
  which every writing code path calls before the write (today `app.operator.create`, right after
  `_validate_model`) -- rejected with `RetentionDaysExceedsMaximumError` before anything is
  written -- and again on every read, via `effective_retention_days` below, which clamps a stored
  value above the cap down to it (the fail-safe for a row written under a higher, earlier cap).
  The cap itself (`Settings.max_retention_days`) is passed in: a plain Pydantic model has no
  access to process configuration. Default: `None` (`DEFAULT_RETENTION_DAYS` applies).

`residency` is deliberately never a field on `TenantSettings` below (see the class docstring): it
lives in `control.tenants`, read through `app.repositories.control.ControlRepository`, not
through `TenantSettingsRepository`. It is listed in this catalog anyway because it is resolved
alongside `content_tracing_opt_in` and `model` from the exact same tenant record
(`app.tenant_record.TenantRecord`; `app.observability.resolve_tenant_tracing`,
`app.residency.resolve_residency_route`, `app.llm.resolve_tenant_chat_model`) and a reader of one
needs to see the other.
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
#
# A tenant may set its own `retention_days` shorter or longer than this default, but never past
# the deployment-wide `Settings.max_retention_days` (#84, ADR-0006; GDPR Art. 5(1)(e), storage
# limitation) -- see `effective_retention_days` below for the read-side clamp, and the
# `retention_days` catalog entry above for where the write-side rejection lives.
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


class RetentionDaysExceedsMaximumError(ValueError):
    """`retention_days` is above the deployment's `Settings.max_retention_days` (#84, ADR-0006;
    GDPR Art. 5(1)(e)). Raised by `TenantSettings.require_retention_within_cap`, the write-side
    half of the cap, before any write happens; the read-side half is `effective_retention_days`
    below, which clamps instead of raising."""


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

    def require_retention_within_cap(self, *, max_days: int) -> None:
        """The write-side half of the retention cap (#84, ADR-0006): refuses, with
        `RetentionDaysExceedsMaximumError`, a `retention_days` above `max_days`
        (`Settings.max_retention_days`, passed in because a plain Pydantic model has no access to
        process configuration). Every code path that writes a tenant's settings calls this before
        the write (today: `app.operator.create.create_tenant`); a stored row that nevertheless
        exceeds a later, lower cap is handled by the read-side clamp, `effective_retention_days`
        below, never by this method. `None` (never set) always passes."""
        if self.retention_days is not None and self.retention_days > max_days:
            raise RetentionDaysExceedsMaximumError(
                f"retention_days={self.retention_days} exceeds this deployment's maximum of "
                f"{max_days} days (MAX_RETENTION_DAYS, see app.config.Settings)"
            )


def effective_retention_days(settings: TenantSettings, *, max_days: int) -> int:
    """The one place that knows the read-side conversation-retention rule (#84, ADR-0006): the
    single function `app.retention.run_retention_job` calls so no second copy of this rule can
    ever drift from it.

    - never set (`settings.retention_days is None`) -- `DEFAULT_RETENTION_DAYS`;
    - set, and at or below `max_days` (`Settings.max_retention_days`) -- the tenant's own value,
      honored as-is;
    - set, but above `max_days` -- clamped down to `max_days`. This is the fail-safe half of the
      cap (#84): `TenantSettings` cannot itself reject a value against a deployment maximum it has
      no access to (see the `retention_days` catalog entry above), and a row may also simply
      predate a later, lower `MAX_RETENTION_DAYS` -- either way, a stored value can never make a
      conversation outlive the deployment's *current* cap. Purely a clamp, never a raise: the
      caller (it knows the tenant) is responsible for logging the one line naming the tenant and
      the stored value when this clamps something.
    """
    if settings.retention_days is None:
        return DEFAULT_RETENTION_DAYS
    return min(settings.retention_days, max_days)
