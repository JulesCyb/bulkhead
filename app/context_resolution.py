"""Context resolution: the one chain from "a tenant id from the path and an authorization header"
to the `RequestContext` a request acts as -- or one typed rejection (#101, spec #91, ADR-0003,
ADR-0005, ADR-0010, ADR-0012).

Glossary (`CONTEXT.md`):

- **Tenant** -- named by the path (`/v1/t/{tenant_id}/...`, ADR-0012) and by nothing else.
- **Identity** -- a person or agent known by (issuer, subject); a token names one.
- **Membership** -- the identity's relation to this tenant; its **Role** is the only role the
  resulting context carries (never a client-supplied one, except under `dev-headers`).
- **Suspension** -- a suspended tenant gets no context at all, from any adapter.
- **Delegation** -- a person's request runs with the person's membership; the means is the agent
  that handles it (`("agent", "assistant")`).
- **Agent identity** -- an agent's own identity acting with no person present; the means is the
  credential that authenticated it (`("credential", <public id>)`).

What this module does, in order, for a bearer token (`resolve_bearer_context`):

1. Parses the `Authorization` header (`Bearer <token>`) -- otherwise 401.
2. Verifies the token and resolves identity and membership through
   `app.token_verifier.verify_tenant_token`, with the verification key and algorithm pinned per
   issuer (`key_source_for`/`algorithm_source_for`: a person's token against the identity
   provider's settings, an agent token against this application's own) -- signature/issuer/expiry
   failures are 401, audience/identity/membership failures are 403.
3. Checks suspension (`app.tenant_suspension.ensure_tenant_not_suspended`) -- otherwise 403.
4. Assigns the means (`actor_context`) and builds the context with a fresh request id.

`AUTH_MODE=dev-headers` (local development only) is its own function, `resolve_dev_headers_context`,
returning the same value type: the identity and roles come from `X-Identity-Id`/`X-Roles`, the
tenant still from the path, suspension is still checked, and the means is still delegation.
`resolve_request_context` selects between the two by `settings.auth_mode`; it is the one function
an adapter calls.

Adapters (the FastAPI dependency `app.deps.get_context` today; the MCP middleware in #102) read
their transport's inputs, call `resolve_request_context` (or `resolve_bearer_context` where only
a bearer token makes sense), and render a `ContextRejection` in their transport's shape: its
`status` is the HTTP status code, its `detail` the only thing a client may see, and a
`FORBIDDEN` rejection is a security event the adapter logs with `reason`, tenant, `issuer`, and
`request_id` -- never the token. Nothing outside this module builds a request's context
(`tests/test_context_resolution.py` greps for it).

`adapter` (`app.token_verifier.ControlPlaneReads`, #100) is the one seam for every control-plane
read the chain makes -- identity, tenant auth settings (issuer and suspension), membership; it
defaults to the real repositories. Tests pass (or install) `tests.conftest.FakeControlPlaneReads`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Final
from uuid import UUID

from app.config import Settings
from app.context import MeansKind, RequestContext
from app.jwt_verifier import KeySource, TokenVerificationError
from app.tenant_suspension import TenantSuspendedError, ensure_tenant_not_suspended
from app.token_verifier import (
    AGENT_IDENTITY_ISSUER,
    AlgorithmSource,
    ControlPlaneReads,
    ResolvedIdentity,
    TenantTokenVerificationError,
    VerificationFailureReason,
    verify_tenant_token,
)

__all__ = [
    "DEV_HEADERS_ISSUER",
    "FORBIDDEN_DETAIL",
    "ContextRejection",
    "RejectionReason",
    "RejectionStatus",
    "ResolvedIdentity",
    "actor_context",
    "algorithm_source_for",
    "key_source_for",
    "parse_bearer_token",
    "resolve_bearer_context",
    "resolve_dev_headers_context",
    "resolve_request_context",
]

# Every forbidden rejection carries exactly this detail, whatever check failed -- the reason is
# discoverable only from the adapter's security-event log line, never the response (ADR-0012).
FORBIDDEN_DETAIL: Final[str] = "Not authorized for this tenant."

# The issuer a dev-headers rejection names in the security-event log -- there is no token.
DEV_HEADERS_ISSUER: Final[str] = "dev-headers"

# Delegation's means (ADR-0005): which agent handles a person's request.
_DELEGATION: Final[tuple[MeansKind, str]] = ("agent", "assistant")


class RejectionStatus(IntEnum):
    """The status class of a rejection, valued as the HTTP status code an adapter answers with --
    one table, so no adapter keeps a second mapping of its own."""

    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403


class RejectionReason(StrEnum):
    """Why a request got no context. The four token reasons carry exactly the values of
    `app.token_verifier.VerificationFailureReason`; the others are this module's own."""

    MISSING_OR_MALFORMED_BEARER = "missing_or_malformed_bearer"
    INVALID_OR_EXPIRED = VerificationFailureReason.INVALID_OR_EXPIRED.value
    AUDIENCE_MISMATCH = VerificationFailureReason.AUDIENCE_MISMATCH.value
    UNKNOWN_IDENTITY = VerificationFailureReason.UNKNOWN_IDENTITY.value
    MISSING_MEMBERSHIP = VerificationFailureReason.MISSING_MEMBERSHIP.value
    TENANT_SUSPENDED = "tenant_suspended"
    DEV_IDENTITY_MISSING = "dev_identity_missing"
    DEV_IDENTITY_INVALID = "dev_identity_invalid"


@dataclass(frozen=True, slots=True)
class ContextRejection:
    """A request that gets no context -- a value, never a transport exception.

    `status`: the HTTP status code to answer with. `reason`: for the operator's log only.
    `detail`: the only text a client may see. `request_id`: correlates the log line.
    `issuer`: the issuer the token was checked against, when known (`DEV_HEADERS_ISSUER` under
    dev-headers; `None` when no issuer could be resolved at all)."""

    status: RejectionStatus
    reason: RejectionReason
    detail: str
    request_id: str
    issuer: str | None = None

    @property
    def is_security_event(self) -> bool:
        """A forbidden rejection is logged by the adapter as one security event; an
        unauthenticated or malformed request is not."""
        return self.status is RejectionStatus.FORBIDDEN


def key_source_for(settings: Settings) -> KeySource:
    """The verification key, pinned per issuer: an agent identity's token
    (`iss == AGENT_IDENTITY_ISSUER`, minted by `app/agent_credential_exchange.py`) only ever
    against this application's own agent-token key; every other issuer only ever against the
    identity provider's `jwt_verification_key`. A single process-wide key (ADR-0003's
    operator-run identity provider), never a network JWKS fetch; a customer-owned identity
    provider needs a JWKS-backed `KeySource` -- override `app.deps.get_key_source`, never make
    `app/jwt_verifier.py` reach the network itself."""

    def _source(issuer: str, kid: str | None) -> str:
        if issuer == AGENT_IDENTITY_ISSUER:
            if settings.agent_token_signing_key is None:
                raise TokenVerificationError("no agent token signing key configured")
            # Asymmetric agent_token_algorithm: verify against the public key (explicit, or
            # derived at Settings construction, see app/config.py), never the private signing
            # key itself. Symmetric (HS*, the default): the signing key is the shared secret.
            if settings.agent_token_verification_key is not None:
                return settings.agent_token_verification_key.get_secret_value()
            return settings.agent_token_signing_key.get_secret_value()
        if settings.jwt_verification_key is None:
            raise TokenVerificationError("no verification key configured")
        return settings.jwt_verification_key.get_secret_value()

    return _source


def algorithm_source_for(settings: Settings) -> AlgorithmSource:
    """The algorithm half of the same per-issuer pinning (algorithm-confusion guard): an agent
    identity's token only ever against `agent_token_algorithm`, every other issuer only ever
    against `jwt_algorithm`. Never derived from the token's own header."""

    def _source(issuer: str) -> tuple[str, ...]:
        if issuer == AGENT_IDENTITY_ISSUER:
            return (settings.agent_token_algorithm,)
        return (settings.jwt_algorithm,)

    return _source


def parse_bearer_token(authorization: str | None) -> str | None:
    """The token of an `Authorization: Bearer <token>` header value, or `None` if the header is
    missing, names another scheme, or carries no token."""
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    return token or None


def _new_context(
    tenant_id: UUID,
    identity_id: UUID,
    roles: frozenset[str],
    means: tuple[MeansKind, str],
    request_id: str | None,
) -> RequestContext:
    # The one construction site of a request's context.
    ctx = RequestContext(
        tenant_id=tenant_id,
        identity_id=identity_id,
        roles=roles,
        request_id=request_id or uuid.uuid4().hex,
    )
    kind, means_id = means
    return ctx.acting_through(kind, means_id)


def actor_context(
    tenant_id: UUID, resolved: ResolvedIdentity, *, request_id: str | None = None
) -> RequestContext:
    """The context a verified token acts as, naming its means (ADR-0005): a person's token is
    delegation (the assistant as the means); an agent identity's token is autonomous use (the
    credential from the token's own `cred` claim as the means)."""
    if resolved.issuer == AGENT_IDENTITY_ISSUER:
        means: tuple[MeansKind, str] = ("credential", resolved.credential_public_id or "unknown")
    else:
        means = _DELEGATION
    return _new_context(
        tenant_id, resolved.identity_id, frozenset({resolved.role}), means, request_id
    )


def _forbidden(reason: RejectionReason, request_id: str, issuer: str | None) -> ContextRejection:
    return ContextRejection(
        status=RejectionStatus.FORBIDDEN,
        reason=reason,
        detail=FORBIDDEN_DETAIL,
        request_id=request_id,
        issuer=issuer,
    )


async def _suspension_rejection(
    tenant_id: UUID, request_id: str, issuer: str | None, adapter: ControlPlaneReads | None
) -> ContextRejection | None:
    """Suspension, checked inside the module so no adapter can hand out a context for a
    suspended tenant (ADR-0010). A tenant with no control-plane row is not suspended."""
    try:
        await ensure_tenant_not_suspended(tenant_id, adapter=adapter)
    except TenantSuspendedError:
        return _forbidden(RejectionReason.TENANT_SUSPENDED, request_id, issuer)
    return None


async def resolve_bearer_context(
    *,
    tenant_id: UUID,
    authorization: str | None,
    settings: Settings,
    adapter: ControlPlaneReads | None = None,
    key_source: KeySource | None = None,
    algorithm_source: AlgorithmSource | None = None,
) -> RequestContext | ContextRejection:
    """The bearer-token chain (module docstring). `key_source`/`algorithm_source` default to
    `key_source_for(settings)`/`algorithm_source_for(settings)`; an adapter passes its own only to
    honour an override (`app.deps.get_key_source` under `dependency_overrides`)."""
    request_id = uuid.uuid4().hex

    token = parse_bearer_token(authorization)
    if token is None:
        return ContextRejection(
            status=RejectionStatus.UNAUTHORIZED,
            reason=RejectionReason.MISSING_OR_MALFORMED_BEARER,
            detail="Missing or malformed bearer token",
            request_id=request_id,
        )

    try:
        resolved = await verify_tenant_token(
            token,
            tenant_id=tenant_id,
            key_source=key_source or key_source_for(settings),
            default_issuer=settings.default_identity_issuer,
            algorithm_source=algorithm_source or algorithm_source_for(settings),
            adapter=adapter,
        )
    except TenantTokenVerificationError as exc:
        reason = RejectionReason(exc.reason.value)
        if exc.reason is VerificationFailureReason.INVALID_OR_EXPIRED:
            return ContextRejection(
                status=RejectionStatus.UNAUTHORIZED,
                reason=reason,
                detail=(
                    "No token issuer configured"
                    if exc.issuer is None
                    else "Invalid or expired token"
                ),
                request_id=request_id,
                issuer=exc.issuer,
            )
        return _forbidden(reason, request_id, exc.issuer)

    rejection = await _suspension_rejection(tenant_id, request_id, resolved.issuer, adapter)
    if rejection is not None:
        return rejection
    return actor_context(tenant_id, resolved, request_id=request_id)


async def resolve_dev_headers_context(
    *,
    tenant_id: UUID,
    identity_header: str | None,
    roles_header: str | None,
    adapter: ControlPlaneReads | None = None,
) -> RequestContext | ContextRejection:
    """`AUTH_MODE=dev-headers` (local development only): the identity from `X-Identity-Id`, the
    roles from `X-Roles` (a development-only convenience with no production equivalent), the
    tenant from the path, suspension checked, the means delegation."""
    request_id = uuid.uuid4().hex
    if not identity_header:
        return ContextRejection(
            status=RejectionStatus.UNAUTHORIZED,
            reason=RejectionReason.DEV_IDENTITY_MISSING,
            detail="X-Identity-Id is missing (AUTH_MODE=dev-headers)",
            request_id=request_id,
            issuer=DEV_HEADERS_ISSUER,
        )
    try:
        identity_id = UUID(identity_header)
    except ValueError:
        return ContextRejection(
            status=RejectionStatus.BAD_REQUEST,
            reason=RejectionReason.DEV_IDENTITY_INVALID,
            detail="Invalid UUID in header",
            request_id=request_id,
            issuer=DEV_HEADERS_ISSUER,
        )
    roles = frozenset(r.strip() for r in (roles_header or "").split(",") if r.strip())

    rejection = await _suspension_rejection(tenant_id, request_id, DEV_HEADERS_ISSUER, adapter)
    if rejection is not None:
        return rejection
    return _new_context(tenant_id, identity_id, roles, _DELEGATION, request_id)


async def resolve_request_context(
    *,
    tenant_id: UUID,
    authorization: str | None,
    settings: Settings,
    dev_identity_id: str | None = None,
    dev_roles: str | None = None,
    adapter: ControlPlaneReads | None = None,
    key_source: KeySource | None = None,
    algorithm_source: AlgorithmSource | None = None,
) -> RequestContext | ContextRejection:
    """The one function an adapter calls: `resolve_dev_headers_context` under
    `AUTH_MODE=dev-headers` (the two `dev_*` values are ignored otherwise), else
    `resolve_bearer_context`."""
    if settings.auth_mode == "dev-headers":
        return await resolve_dev_headers_context(
            tenant_id=tenant_id,
            identity_header=dev_identity_id,
            roles_header=dev_roles,
            adapter=adapter,
        )
    return await resolve_bearer_context(
        tenant_id=tenant_id,
        authorization=authorization,
        settings=settings,
        adapter=adapter,
        key_source=key_source,
        algorithm_source=algorithm_source,
    )
