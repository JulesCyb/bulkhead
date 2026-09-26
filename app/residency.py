"""Residency: one module, one object (ADR-0008, spec A4 / #94, #110).

`ResidencyAllowList` is the allow-list itself -- loaded from `config/residency.toml` (or an
in-memory dict, for tests), validated once (required keys, types, and the no-shared-hosts rule),
and the single place that answers "is this residency known, and what is its route": `route_for`
(the model/gateway host patterns, embedding endpoint, and trace sink for a residency),
`alias_for` (the bare gateway alias for a tenant's chosen model name in a residency -- strips a
`<provider>:` prefix), `model_aliases` (a residency's whole alias list, used by gateway
provisioning), `residencies` (every configured residency), and `check_deployment` (the
deployment's own self-check: its configured gateway host, default model, and trace sink against
its own residency's route -- what `app.startup_checks.run_startup_checks` calls).

**One exception type for every fail-closed lookup or validation this module performs:
`ResidencyUnresolved`.** A malformed, missing, or unsafe (two residencies sharing a host) config
file is `ResidencyConfigError`, kept as a subclass of `ResidencyUnresolved` rather than a sibling
type -- both a bad file and an unknown residency are the same class of failure ("this allow-list
cannot answer that"), and a caller that only ever wants to fail closed can catch the one base
type. Every message names the reason and, where relevant, the offending residency/host/model, so
an operator can fix it without reading this module's source.

`Settings.residency_allow_list` (`app/config.py`) holds one instance, built from the configured
path at construction (never at import) -- this module's own `resolve_residency_route` (the
per-tenant, per-request resolver: a function of the tenant record, `app.tenant_record`, composed
with the tenant's gateway credential; #105) and
`app.startup_checks.run_startup_checks` both read that instance rather than a module-level
global. `app/llm.py` (`validate_model_for_residency`, via `alias_for`), `app/embeddings.py` (via
`resolve_residency_route`), `app/observability.py` (`setup_observability`/
`instrumentation_capabilities`, via `residencies`/`route_for`), and `app/operator/create.py`
(`_validate_residency`/`_validate_model`, via `route_for`/`alias_for`) also read it from
`Settings` -- every one of them calls this object's own methods rather than re-implementing the
lookup (spec A4 / #111): this module is the only place "is this residency known, and what is its
route/alias" is ever answered.

Deferred imports inside functions below (`app.config.get_settings`, `app.gateway_credentials`)
are deliberate, not an oversight: `app.config.Settings` holds a `ResidencyAllowList` field, so
`app.config` imports this module at its own top level; importing those back at this module's top
level would be a real import cycle, not just an untidy one.
"""

from __future__ import annotations

import fnmatch
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, SecretStr

from app.tenant_record import TenantRecord

if TYPE_CHECKING:
    from app.config import Settings

# Default location of the residency allow-list file, relative to the repository root (this file
# lives at <repo>/app/residency.py). `Settings.residency_config_path`
# (`RESIDENCY_CONFIG_PATH` env var) overrides it at construction; `ResidencyAllowList.load`'s own
# default (used by anything that loads the allow-list outside of `Settings`, e.g. a docs-generation
# script) falls back to this same path.
DEFAULT_RESIDENCY_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "residency.toml"


class ResidencyRoute(BaseModel):
    """The endpoints one residency may reach on a content-bearing path."""

    model_host_patterns: tuple[str, ...]
    embedding_endpoint: str
    trace_sink_host: str


class ResidencyUnresolved(Exception):
    """The one fail-closed exception this module raises, for every lookup or validation
    `ResidencyAllowList` performs and for `resolve_residency_route` below: an unknown residency,
    a model outside a residency's alias list, a deployment endpoint outside its own residency's
    allow-list, or a tenant whose own `control.tenants.residency` cannot be resolved to a route.
    Never masked, never a fallback to a default or another residency's route.
    """


class ResidencyConfigError(ResidencyUnresolved):
    """The residency allow-list configuration itself is missing, unreadable, malformed, empty, or
    lets two residencies reach the same host -- a subclass of `ResidencyUnresolved` (both are "this
    allow-list cannot answer that", one from a bad source file, the other from a bad lookup
    against a good one) rather than a second, unrelated exception type. Always names the offending
    file and the exact problem. Raised at `ResidencyAllowList.load`/`from_data` time -- a broken
    allow-list must stop the process from ever starting, never be discovered later at request
    time.
    """


def _host(endpoint: str) -> str:
    """The bare hostname of a URL or a bare host string alike (`langfuse_host` is configured
    without a scheme; `litellm_base_url` and the allow-list's `embedding_endpoint` are full
    URLs)."""
    parsed = urlparse(endpoint if "://" in endpoint else f"//{endpoint}")
    return (parsed.netloc or endpoint).split(":", 1)[0]


def _host_matches_any(host: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(host, pattern) for pattern in patterns)


class ResidencyAllowList:
    """The residency allow-list (ADR-0008): which residencies exist, the endpoints each may
    reach, and the gateway model aliases each may use. Treated as immutable once built --
    nothing here mutates `self` after `__init__`.
    """

    def __init__(
        self,
        allow_list: dict[str, ResidencyRoute],
        model_allow_list: dict[str, tuple[str, ...]],
        *,
        source: str = "<in-memory>",
    ) -> None:
        self._allow_list = dict(allow_list)
        self._model_allow_list = dict(model_allow_list)
        self.source = source

    @classmethod
    def load(cls, path: str | Path | None = None) -> ResidencyAllowList:
        """Loads and validates the residency allow-list from a TOML file (ADR-0008).

        `path` defaults to `DEFAULT_RESIDENCY_CONFIG_PATH`; `Settings`'s own default construction
        passes its own `residency_config_path` (`RESIDENCY_CONFIG_PATH` env var) instead. Raises
        `ResidencyConfigError` if the file is missing, unreadable, or not valid TOML; delegates
        the rest of the validation (required keys, types, the no-shared-hosts rule) to
        `from_data`.
        """
        config_path = Path(path) if path is not None else DEFAULT_RESIDENCY_CONFIG_PATH
        try:
            raw = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ResidencyConfigError(
                f"residency configuration file not found or unreadable: {config_path} ({exc}). "
                "Set RESIDENCY_CONFIG_PATH, or restore config/residency.toml "
                "(see docs/residency.md)."
            ) from exc
        try:
            data = tomllib.loads(raw)
        except tomllib.TOMLDecodeError as exc:
            raise ResidencyConfigError(
                f"residency configuration file {config_path} is not valid TOML: {exc}"
            ) from exc
        return cls.from_data(data, source=str(config_path))

    @classmethod
    def from_data(cls, data: dict, *, source: str = "<in-memory>") -> ResidencyAllowList:
        """Builds and validates a `ResidencyAllowList` from an already-parsed dict (the shape
        `tomllib.loads` returns: `{"residency": {"<name>": {...}, ...}}`) -- the seam a test uses
        to build one without a file on disk. Runs every validation `load` runs against a file:
        each `[residency.<name>]` table must define `model_host_patterns` (a non-empty list of
        glob host patterns), `embedding_endpoint`, and `trace_sink_host` (both non-empty strings);
        `models` (a list of gateway alias names) is optional. Rejects a file in which two
        residencies name the same host or pattern (in `model_host_patterns`, `embedding_endpoint`'s
        host, or `trace_sink_host`) -- residencies that can reach the same host are not actually
        separated, whatever their allow-lists claim. Raises `ResidencyConfigError` for every
        failure mode above -- never a partial or best-effort allow-list, and never a silent
        fallback to an empty or default configuration.
        """
        residencies = data.get("residency")
        if not isinstance(residencies, dict) or not residencies:
            raise ResidencyConfigError(
                f"{source}: must define at least one [residency.<name>] table"
            )

        allow_list: dict[str, ResidencyRoute] = {}
        model_allow_list: dict[str, tuple[str, ...]] = {}
        for name, entry in residencies.items():
            if not isinstance(entry, dict):
                raise ResidencyConfigError(
                    f"{source}: [residency.{name}] must be a table, got {type(entry).__name__}"
                )
            required_keys = ("model_host_patterns", "embedding_endpoint", "trace_sink_host")
            missing = [k for k in required_keys if k not in entry]
            if missing:
                raise ResidencyConfigError(
                    f"{source}: [residency.{name}] is missing required key(s): {missing}"
                )
            host_patterns = entry["model_host_patterns"]
            if (
                not isinstance(host_patterns, list)
                or not host_patterns
                or not all(isinstance(p, str) and p for p in host_patterns)
            ):
                raise ResidencyConfigError(
                    f"{source}: [residency.{name}].model_host_patterns must be a non-empty "
                    "list of non-empty strings"
                )
            embedding_endpoint = entry["embedding_endpoint"]
            trace_sink_host = entry["trace_sink_host"]
            if not isinstance(embedding_endpoint, str) or not embedding_endpoint:
                raise ResidencyConfigError(
                    f"{source}: [residency.{name}].embedding_endpoint must be a non-empty string"
                )
            if not isinstance(trace_sink_host, str) or not trace_sink_host:
                raise ResidencyConfigError(
                    f"{source}: [residency.{name}].trace_sink_host must be a non-empty string"
                )
            models = entry.get("models", [])
            if not isinstance(models, list) or not all(isinstance(m, str) and m for m in models):
                raise ResidencyConfigError(
                    f"{source}: [residency.{name}].models must be a list of non-empty strings"
                )

            allow_list[name] = ResidencyRoute(
                model_host_patterns=tuple(host_patterns),
                embedding_endpoint=embedding_endpoint,
                trace_sink_host=trace_sink_host,
            )
            if models:
                model_allow_list[name] = tuple(models)

        # Fail closed if two residencies could ever reach the same host: that defeats the entire
        # point of a per-residency allow-list. Compared as literal strings (a glob pattern is
        # only ever equal to itself here), which is enough to catch the copy-paste mistake this
        # guards against -- two residencies naming the exact same provider host.
        host_owner: dict[str, str] = {}
        for name, route in allow_list.items():
            embedding_host = route.embedding_endpoint.split("//", 1)[-1].split("/", 1)[0]
            for host in (*route.model_host_patterns, embedding_host, route.trace_sink_host):
                owner = host_owner.get(host)
                if owner is not None and owner != name:
                    raise ResidencyConfigError(
                        f"{source}: host/pattern {host!r} is allow-listed for both "
                        f"{owner!r} and {name!r} -- residencies must never share a host"
                    )
                host_owner[host] = name

        return cls(allow_list, model_allow_list, source=source)

    @property
    def residencies(self) -> tuple[str, ...]:
        """Every configured residency, in file order."""
        return tuple(self._allow_list)

    def route_for(self, residency: str) -> ResidencyRoute:
        """The allow-listed route for `residency`. Raises `ResidencyUnresolved` if `residency`
        has no entry -- never a default or another residency's route."""
        try:
            return self._allow_list[residency]
        except KeyError:
            raise ResidencyUnresolved(
                f"residency {residency!r} has no entry in the residency allow-list "
                f"(configured residencies: {sorted(self._allow_list)})"
            ) from None

    def model_aliases(self, residency: str) -> tuple[str, ...]:
        """`residency`'s whole gateway model alias list -- `()` if it has none configured (never
        an empty tuple standing in for "anything goes"; used by gateway provisioning to restrict a
        freshly minted credential's usable models to its residency)."""
        return self._model_allow_list.get(residency, ())

    def alias_for(self, residency: str, model_name: str) -> str:
        """The bare gateway alias for `model_name` under `residency` -- strips a `<provider>:`
        prefix, if any (the gateway's model_list only knows bare names). Raises
        `ResidencyUnresolved` if the bare name is not on `residency`'s model allow-list, before
        any client is built and before any network call is attempted (ADR-0009)."""
        bare = model_name.split(":", 1)[1] if ":" in model_name else model_name
        allowed = self.model_aliases(residency)
        if bare not in allowed:
            raise ResidencyUnresolved(
                f"model {model_name!r} is not on the allow-list for residency {residency!r} "
                f"(allowed: {sorted(allowed)})"
            )
        return bare

    def check_deployment(self, settings: Settings) -> None:
        """The deployment's own self-check (ADR-0008/ADR-0009): raises `ResidencyUnresolved`
        before the process may accept its first request or tool call when, for the deployment's
        own residency (`settings.residency`) --

        - this allow-list has no entry for it at all (defensive: `Settings` construction already
          rejects an unknown residency against this same object, but this method never trusts
          that as its only guard);
        - the configured model/gateway host (`settings.litellm_base_url`) is not on its allow-listed
          model host patterns;
        - the deployment's default model (`settings.llm_model`) is not on its model allow-list;
        - the embedding endpoint the deployment will actually reach (the same gateway host, the
          only path `app.embeddings` ever calls) is not its allow-listed embedding endpoint, and
          not otherwise covered by its model host patterns;
        - a configured trace sink (`settings.langfuse_host`) is not its allow-listed trace sink
          host.

        A fully valid, allow-listed configuration returns without raising. This is exactly what
        `app.startup_checks.run_startup_checks` calls.
        """
        residency = settings.residency
        route = self.route_for(residency)

        # Unconditional (ADR-0009): `settings.litellm_base_url` is guaranteed truthy here --
        # `Settings` refuses to construct without it -- so there is no "gateway not configured"
        # branch to skip past.
        gateway_host = _host(settings.litellm_base_url)  # type: ignore[arg-type]
        if not _host_matches_any(gateway_host, route.model_host_patterns):
            raise ResidencyUnresolved(
                f"residency {residency!r}: model/gateway host {gateway_host!r} "
                f"(LITELLM_BASE_URL={settings.litellm_base_url!r}) is not on its allow-listed "
                f"host patterns {route.model_host_patterns}"
            )
        try:
            self.alias_for(residency, settings.llm_model)
        except ResidencyUnresolved as exc:
            raise ResidencyUnresolved(
                f"residency {residency!r}: default model {settings.llm_model!r} is not on its "
                f"model allow-list -- {exc}"
            ) from exc

        # Embeddings are only ever reachable through this same gateway host (ADR-0009,
        # app/embeddings.py refuses to build a client with no gateway configured).
        embedding_host = gateway_host
        if embedding_host != _host(route.embedding_endpoint) and not _host_matches_any(
            embedding_host, route.model_host_patterns
        ):
            raise ResidencyUnresolved(
                f"residency {residency!r}: embedding endpoint host {embedding_host!r} is not "
                f"its allow-listed embedding endpoint {route.embedding_endpoint!r}"
            )

        if settings.langfuse_host:
            trace_host = _host(settings.langfuse_host)
            if trace_host != route.trace_sink_host:
                raise ResidencyUnresolved(
                    f"residency {residency!r}: trace sink host {trace_host!r} "
                    f"(LANGFUSE_HOST={settings.langfuse_host!r}) is not its allow-listed trace "
                    f"sink {route.trace_sink_host!r}"
                )


class ResolvedResidencyRoute(BaseModel):
    """Everything a content-bearing call needs for one tenant, resolved together.

    Bundling the residency's route and the tenant's own gateway credential into one immutable
    object -- rather than returning them separately for a caller to recombine -- is what makes
    "a credential and an endpoint from two different tenants are never combined" true by
    construction: there is no seam at which two `ResolvedResidencyRoute` values, or parts of
    them, could be mixed.
    """

    model_config = ConfigDict(frozen=True)

    residency: str
    route: ResidencyRoute
    gateway_credential: SecretStr


def _settings_or_default(settings: Settings | None) -> Settings:
    # Deferred import (see module docstring): `app.config` imports this module at its own top
    # level, so importing `get_settings` back at this module's top level would be a real cycle.
    if settings is not None:
        return settings
    from app.config import get_settings

    return get_settings()


def tenant_residency_route(
    record: TenantRecord, *, settings: Settings | None = None
) -> tuple[str, ResidencyRoute]:
    """The tenant's residency and its allow-listed route, from the record alone -- no credential.

    Raises `ResidencyUnresolved` if the record carries no residency or one the allow-list does
    not know (ADR-0008: fail closed, never a default or another residency's route). The first
    step of `resolve_residency_route` below, and of `app.llm.resolve_tenant_chat_model`, which
    must validate the model name against this residency *before* any credential is read
    (ADR-0009)."""
    s = _settings_or_default(settings)
    residency = record.residency
    if not residency:
        raise ResidencyUnresolved(
            f"tenant {record.tenant_id} has no usable residency (control.tenants.residency = "
            f"{residency!r})"
        )
    try:
        route = s.residency_allow_list.route_for(residency)
    except ResidencyUnresolved as exc:
        raise ResidencyUnresolved(
            f"tenant {record.tenant_id} has no usable residency (control.tenants.residency = "
            f"{residency!r}); configured residencies: "
            f"{sorted(s.residency_allow_list.residencies)}"
        ) from exc
    return residency, route


def resolve_residency_route(
    record: TenantRecord, *, settings: Settings | None = None
) -> ResolvedResidencyRoute:
    """The single call site every content-bearing path resolves its route from (ADR-0008) -- a
    function of the tenant record and settings, with no database read of its own (#105): the
    residency and the gateway credential alias were read once, with the rest of the record, when
    the request's context was resolved (`app.tenant_record`).

    Composes three facts that must never be mixed across tenants: the tenant's own residency
    (`record.residency`), the allow-listed route for that residency (`settings.residency_allow_
    list`, read from `get_settings()` when `settings` is not given), and the tenant's own gateway
    credential (the file `record.gateway_credential_alias` names). Raises `ResidencyUnresolved` if
    the residency is missing, unknown, or absent from the allow-list -- before the credential is
    read -- and propagates `GatewayCredentialUnavailable`
    (`app.gateway_credentials.GatewayCredentialUnavailable`) unchanged if the alias is missing or
    its file is missing/empty -- both are fail-closed errors, never a fallback.
    """
    s = _settings_or_default(settings)
    residency, route = tenant_residency_route(record, settings=s)

    # Deferred import (see module docstring): `app.gateway_credentials` imports `app.config` at
    # its own top level, so importing it back at this module's top level would be a real cycle.
    from app.gateway_credentials import resolve_gateway_credential

    return ResolvedResidencyRoute(
        residency=residency,
        route=route,
        gateway_credential=resolve_gateway_credential(record, settings=s),
    )


__all__ = [
    "DEFAULT_RESIDENCY_CONFIG_PATH",
    "ResidencyAllowList",
    "ResidencyConfigError",
    "ResidencyRoute",
    "ResidencyUnresolved",
    "ResolvedResidencyRoute",
    "resolve_residency_route",
    "tenant_residency_route",
]
