"""Per-tenant gateway credential resolution (Spec 7 / #52, ADR-0009, ADR-0011).

Distinct from `app.config.Settings`: a gateway credential is per-tenant and must be resolved on
every request from the control plane's own alias -- carried on the tenant record
(`app.tenant_record.TenantRecord.gateway_credential_alias`, read once per request, #104/#105) --
and a fresh disk read, never baked in once at process startup like a Settings field is. Nothing
here is cached across tenants, or across calls for the same tenant -- a rotated file is picked up
on the very next resolution. This is
the shared seam every later per-tenant model/embedding client (and the future credential-minting
call) is built on.

Two failure modes are collapsed into one typed error, `GatewayCredentialUnavailable`, rather
than a silent empty credential or a raw `OSError`/`KeyError` bubbling up: a tenant with no alias
recorded yet, and a tenant whose alias names a file that is missing (or empty) under the
configured directory.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr

from app.config import Settings, get_settings
from app.tenant_record import TenantRecord


class GatewayCredentialUnavailable(Exception):
    """A tenant's gateway credential could not be resolved.

    Raised for exactly two reasons: no `gateway_credential_alias` is recorded for the tenant in
    the control plane yet, or that alias names a secret file that is missing or empty under the
    configured directory. Never raised for, and never masks, any other kind of failure.
    """


def read_gateway_credential(alias: str, *, settings: Settings | None = None) -> SecretStr:
    """Read `alias`'s secret file fresh from `Settings.gateway_credentials_dir`.

    Always reads the file from disk -- nothing is cached here, so a redeploy that rotates the
    file (ADR-0011) is picked up on the very next call. Raises `GatewayCredentialUnavailable` if
    the file is missing or empty rather than letting a raw `OSError` escape.
    """
    directory = Path((settings or get_settings()).gateway_credentials_dir)
    path = directory / alias
    try:
        content = path.read_text().strip()
    except OSError as exc:
        raise GatewayCredentialUnavailable(
            f"gateway credential file for alias {alias!r} is missing under {directory}"
        ) from exc
    if not content:
        raise GatewayCredentialUnavailable(
            f"gateway credential file for alias {alias!r} under {directory} is empty"
        )
    return SecretStr(content)


def resolve_gateway_credential(
    record: TenantRecord, *, settings: Settings | None = None
) -> SecretStr:
    """The tenant's gateway credential end to end: the alias its record carries, then a fresh
    read of that alias's secret file. No database read of its own (#105): the alias was read with
    the rest of the tenant record. Raises `GatewayCredentialUnavailable` if the record has no
    alias or the file is missing/empty. Two tenants with different aliases always resolve to two
    different `SecretStr` values; nothing here is shared or cached between them.
    """
    alias = record.gateway_credential_alias
    if not alias:
        raise GatewayCredentialUnavailable(
            f"tenant {record.tenant_id} has no gateway credential alias recorded in the control "
            "plane"
        )
    return read_gateway_credential(alias, settings=settings)
