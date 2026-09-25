"""Shared tenant-token verification (#44, ADR-0003, ADR-0012, ADR-0005): the signature/issuer/
expiry check (`app/jwt_verifier.py`), the tenant-audience check, and the identity/membership
resolution factored out of `app/deps.py`'s `AUTH_MODE=jwt` branch into one module any caller can
reuse. Today that caller is the HTTP request-context dependency; Spec 6's MCP transport is meant
to call this same function instead of writing a second copy of the check.

Deliberately excluded: tenant suspension. That is not one of the checks this module owns -- each
caller enforces it itself, wherever it resolves a context (issue #69), rather than having it baked
into the one shared verification step.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from app.context import RequestContext
from app.db.session import control_session, tenant_session
from app.jwt_verifier import KeySource, TokenVerificationError, verify_token
from app.repositories.control import IdentityRepository, TenantAuthSettingsRepository
from app.repositories.memberships import MembershipRepository


class VerificationFailureReason(StrEnum):
    """Every categorized way a tenant-scoped token can fail to resolve to an identity and
    membership. `INVALID_OR_EXPIRED` covers a bad signature, an expired token, and no
    verification key/issuer configured for the tenant -- indistinguishable from a caller's point
    of view, and mapped by callers to 401. The rest are mapped by callers to 403."""

    INVALID_OR_EXPIRED = "invalid_or_expired"
    AUDIENCE_MISMATCH = "audience_mismatch"
    UNKNOWN_IDENTITY = "unknown_identity"
    MISSING_MEMBERSHIP = "missing_membership"


class TenantTokenVerificationError(Exception):
    """Raised with exactly one `VerificationFailureReason` -- callers key their status code and
    security-event log off `.reason`. `.issuer` is the issuer the token was checked against, when
    known (None only for `INVALID_OR_EXPIRED` raised before an issuer could be resolved at all).
    The exception message is for logs only -- never echo it back to a client; the response body
    must stay generic (ADR-0012)."""

    def __init__(self, reason: VerificationFailureReason, *, issuer: str | None = None) -> None:
        self.reason = reason
        self.issuer = issuer
        super().__init__(reason.value)


@dataclass(frozen=True, slots=True)
class ResolvedIdentity:
    """What a token resolves to, once verified against one specific tenant."""

    identity_id: UUID
    role: str
    issuer: str


async def verify_tenant_token(
    token: str,
    *,
    tenant_id: UUID,
    key_source: KeySource,
    default_issuer: str | None,
    algorithms: tuple[str, ...],
) -> ResolvedIdentity:
    """Verify `token` against `tenant_id` and resolve it to an identity and its membership.

    Order: signature/issuer/expiry -> the token's audience must equal `tenant_id` -> the
    (issuer, subject) the token names must be a known identity -> that identity must have a
    membership in `tenant_id` (looked up inside that tenant's own context -- no RLS bypass, and no
    separate "does this tenant exist" query: no membership row means `MISSING_MEMBERSHIP`,
    whether the identity truly isn't a member or the tenant simply doesn't exist).

    Raises `TenantTokenVerificationError` with a single categorized reason on the first check
    that fails; returns the resolved identity and role on success. Never checks suspension --
    see module docstring.
    """
    async with control_session() as session:
        auth_settings = await TenantAuthSettingsRepository().get(
            session, tenant_id=tenant_id, default_issuer=default_issuer
        )
    expected_issuer = auth_settings.issuer if auth_settings else default_issuer
    if not expected_issuer:
        raise TenantTokenVerificationError(VerificationFailureReason.INVALID_OR_EXPIRED)

    try:
        claims = verify_token(
            token, key_source=key_source, expected_issuer=expected_issuer, algorithms=algorithms
        )
    except TokenVerificationError:
        raise TenantTokenVerificationError(
            VerificationFailureReason.INVALID_OR_EXPIRED, issuer=expected_issuer
        ) from None

    if claims.audience != str(tenant_id):
        raise TenantTokenVerificationError(
            VerificationFailureReason.AUDIENCE_MISMATCH, issuer=expected_issuer
        )

    async with control_session() as session:
        identity = await IdentityRepository().find_by_issuer_and_subject(
            session, issuer=expected_issuer, subject=claims.subject
        )
    if identity is None:
        raise TenantTokenVerificationError(
            VerificationFailureReason.UNKNOWN_IDENTITY, issuer=expected_issuer
        )

    preliminary_ctx = RequestContext(
        tenant_id=tenant_id, identity_id=identity.id, roles=frozenset()
    )
    async with tenant_session(preliminary_ctx) as session:
        role = await MembershipRepository().get_role(
            session, preliminary_ctx, identity_id=identity.id
        )
    if role is None:
        raise TenantTokenVerificationError(
            VerificationFailureReason.MISSING_MEMBERSHIP, issuer=expected_issuer
        )

    return ResolvedIdentity(identity_id=identity.id, role=role, issuer=expected_issuer)
