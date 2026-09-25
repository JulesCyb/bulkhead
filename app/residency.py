"""Per-tenant residency route resolution (Spec 8 / #60, ADR-0008).

A tenant's `control.tenants.residency` (operator-owned, ADR-0008) names a jurisdiction (for example
`eu`). This module is the one place that turns that setting into an actual route: the allow-listed
model/gateway hosts, embedding endpoint, and trace sink for that residency
(`app.config.RESIDENCY_ALLOW_LIST`), composed with that same tenant's own gateway credential
(`app.gateway_credentials`, #52). Model routing (#54) and embedding client construction (#61) are
both meant to call `resolve_residency_route` for this and build no parallel per-tenant route of
their own -- so two tenants in different residencies can never be handed each other's route, and a
credential from one tenant can never end up paired with an endpoint resolved for another.

Reads the tenant's own row through the same tenant-scoped, RLS-filtered session every repository
uses (`tenant_session(ctx)`, see `app/db/session.py`) -- never a cross-tenant control-plane
session. A residency that is unset, not a string, or absent from the allow-list is a fail-closed
`ResidencyUnresolved`, never a fallback to a default route or another tenant's jurisdiction, per
ADR-0008.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import RESIDENCY_ALLOW_LIST, ResidencyRoute, Settings
from app.context import RequestContext
from app.gateway_credentials import resolve_gateway_credential
from app.repositories.control import ControlRepository


class ResidencyUnresolved(Exception):
    """A tenant's residency could not be resolved to an allow-listed route.

    Raised for exactly these reasons, never masked and never papered over with a default:
    `control.tenants.residency` is unset (no row, or null), it is set but not a
    recognized value, or it names a residency absent from `RESIDENCY_ALLOW_LIST`. No call site
    of this module falls back to a default residency, another tenant's route, or the
    deployment-wide `Settings.residency_route`.
    """


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


async def _read_tenant_residency(session: AsyncSession, ctx: RequestContext) -> str | None:
    """The caller's own residency from the control plane (`control.tenants.residency`, 0013).

    An operator-owned fact: `app` can read it through `control.tenants_view` (RLS-scoped to the
    caller's tenant) but never write it, so a tenant's own request cannot move itself to another
    jurisdiction.
    """
    return await ControlRepository().get_residency(session, ctx)


async def resolve_residency_route(
    session: AsyncSession, ctx: RequestContext, *, settings: Settings | None = None
) -> ResolvedResidencyRoute:
    """The single call site every content-bearing path resolves its route from (ADR-0008).

    Composes three facts that must never be mixed across tenants: the tenant's own residency
    setting, the allow-listed route for that residency, and the tenant's own gateway credential.
    Raises `ResidencyUnresolved` if the residency is missing, unknown, or absent from the
    allow-list, and propagates `GatewayCredentialUnavailable`
    (`app.gateway_credentials.GatewayCredentialUnavailable`) unchanged if the credential itself
    cannot be resolved -- both are fail-closed errors, never a fallback.
    """
    residency = await _read_tenant_residency(session, ctx)
    if not residency or residency not in RESIDENCY_ALLOW_LIST:
        raise ResidencyUnresolved(
            f"tenant {ctx.tenant_id} has no usable residency (control.tenants.residency = "
            f"{residency!r}); configured residencies: {sorted(RESIDENCY_ALLOW_LIST)}"
        )
    gateway_credential = await resolve_gateway_credential(session, ctx, settings=settings)
    return ResolvedResidencyRoute(
        residency=residency,
        route=RESIDENCY_ALLOW_LIST[residency],
        gateway_credential=gateway_credential,
    )
