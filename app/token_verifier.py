"""Shared tenant-token verification (#44, ADR-0003, ADR-0012, ADR-0005): the signature/issuer/
expiry check (`app/jwt_verifier.py`), the tenant-audience check, and the identity/membership
resolution factored out of `app/deps.py`'s `AUTH_MODE=jwt` branch into one module any caller can
reuse. Today that caller is the HTTP request-context dependency; Spec 6's MCP transport is meant
to call this same function instead of writing a second copy of the check.

Deliberately excluded: tenant suspension. That is not one of the checks this module owns -- each
caller enforces it itself, wherever it resolves a context (issue #69), rather than having it baked
into the one shared verification step.

**Gap fix (Spec 6, closing the loop between #46/#47 and this module).** An agent identity's token
(minted by `app/agent_credential_exchange.py`) is *not* governed by a tenant's own human-IdP
`identity_issuer` setting: `control.create_agent_identity` (migration 0032) synthesizes every
agent identity's issuer as the fixed literal `AGENT_IDENTITY_ISSUER` below, the same constant the
exchange module signs with. This module peeks at a presented token's own (unverified) `iss` claim
before deciding which issuer/key pair to check it against: `AGENT_IDENTITY_ISSUER` routes to the
agent-token issuer directly (no tenant auth-settings lookup -- an agent token is tenant-independent
by construction), and anything else falls back to the tenant's own configured issuer exactly as
before. The peek is never trusted on its own: `verify_token` re-checks the real `iss` claim against
whichever issuer this picks, under signature, so a forged `iss` that does not match its own
signature still fails closed the same way it always did.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final
from uuid import UUID

import jwt as _pyjwt

from app.context import RequestContext
from app.db.session import control_session, tenant_session
from app.jwt_verifier import KeySource, TokenVerificationError, verify_token
from app.repositories.control import IdentityRepository, TenantAuthSettingsRepository
from app.repositories.memberships import MembershipRepository

# (issuer) -> the algorithm(s) a token claiming that issuer may be verified with -- the algorithm
# half of the same per-issuer pinning `KeySource` does for the verification key (see
# `app.deps.get_key_source` / `get_algorithm_source`, the two real implementations both the HTTP
# API and the MCP transport use). Never trusts the token's own header for this: `verify_token`
# always passes whatever this resolves to as an explicit allow-list to `jwt.decode`, so a token
# cannot pick its own algorithm, and a token minted under one issuer's algorithm/key can never be
# checked against another issuer's (the algorithm-confusion guard, review finding Spec 6).
AlgorithmSource = Callable[[str], tuple[str, ...]]

# The fixed issuer `control.create_agent_identity` (migration 0032) synthesizes for every agent
# identity, and the issuer `app/agent_credential_exchange.py` mints agent tokens under. Never a
# tenant's own (human-IdP) issuer -- an agent token is tenant-independent by construction, the
# same credential-issuing tenant is instead enforced via the token's audience (below).
AGENT_IDENTITY_ISSUER: Final[str] = "agent"


def _peek_unverified_issuer(token: str) -> str | None:
    """Read the `iss` claim without verifying signature/expiry -- used only to pick which
    issuer/key pair to check the token against for real (see module docstring). Never used to
    make a security decision by itself: `verify_token` re-verifies `iss` under signature
    immediately afterward, so a token whose real `iss` disagrees with what this peek returned
    (or whose signature does not match the key this later resolves to) still fails closed."""
    try:
        claims = _pyjwt.decode(
            token,
            options={
                "verify_signature": False,
                "verify_exp": False,
                "verify_aud": False,
                "verify_iss": False,
            },
        )
    except _pyjwt.PyJWTError:
        return None
    issuer = claims.get("iss")
    return issuer if isinstance(issuer, str) else None


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
    """What a token resolves to, once verified against one specific tenant.

    `credential_public_id` is set only for an agent-identity token (issuer ==
    `AGENT_IDENTITY_ISSUER`) that carries the `cred` claim `app/agent_credential_exchange.py`
    embeds -- the credential that authenticated it, for a caller (the MCP transport, #49) that
    needs to name it as `RequestContext`'s means. `None` for a person's token, which has no
    credential to name."""

    identity_id: UUID
    role: str
    issuer: str
    credential_public_id: str | None = None


async def verify_tenant_token(
    token: str,
    *,
    tenant_id: UUID,
    key_source: KeySource,
    default_issuer: str | None,
    algorithm_source: AlgorithmSource,
) -> ResolvedIdentity:
    """Verify `token` against `tenant_id` and resolve it to an identity and its membership.

    Order: signature/issuer/expiry -> the token's audience must equal `tenant_id` -> the
    (issuer, subject) the token names must be a known identity -> that identity must have a
    membership in `tenant_id` (looked up inside that tenant's own context -- no RLS bypass, and no
    separate "does this tenant exist" query: no membership row means `MISSING_MEMBERSHIP`,
    whether the identity truly isn't a member or the tenant simply doesn't exist).

    `algorithm_source(expected_issuer)` -- called only once `expected_issuer` is resolved (below)
    -- returns the algorithm(s) that issuer's tokens are allowed to verify against, exactly the
    same per-issuer pinning `key_source` already does for the key (algorithm-confusion guard,
    review finding Spec 6): a human issuer never gets checked against the agent algorithm/key, or
    vice versa, and neither ever accepts an algorithm the token's own header names.

    Raises `TenantTokenVerificationError` with a single categorized reason on the first check
    that fails; returns the resolved identity and role on success. Never checks suspension --
    see module docstring.
    """
    if _peek_unverified_issuer(token) == AGENT_IDENTITY_ISSUER:
        # An agent identity's token: tenant-independent issuer (see module docstring) -- skip the
        # tenant auth-settings lookup entirely, since it has nothing to say about this issuer.
        expected_issuer = AGENT_IDENTITY_ISSUER
    else:
        async with control_session() as session:
            auth_settings = await TenantAuthSettingsRepository().get(
                session, tenant_id=tenant_id, default_issuer=default_issuer
            )
        expected_issuer = auth_settings.issuer if auth_settings else default_issuer
        if not expected_issuer:
            raise TenantTokenVerificationError(VerificationFailureReason.INVALID_OR_EXPIRED)

    algorithms = algorithm_source(expected_issuer)
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

    return ResolvedIdentity(
        identity_id=identity.id,
        role=role,
        issuer=expected_issuer,
        credential_public_id=claims.extra.get("cred"),
    )
