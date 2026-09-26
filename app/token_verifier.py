"""Shared tenant-token verification (#44, ADR-0003, ADR-0012, ADR-0005): the signature/issuer/
expiry check (`app/jwt_verifier.py`), the tenant-audience check, and the identity/membership
resolution factored out of `app/deps.py`'s `AUTH_MODE=jwt` branch into one module any caller can
reuse. Its caller is `app/context_resolution.py` (#101), the one chain every adapter (HTTP, and the
MCP transport) resolves a request's context through.

Tenant suspension is decided here only when a caller asks for the tenant record
(`read_tenant_record=True`, #104 -- `app/context_resolution.py` always does): the record is read
right after the token's signature/issuer/expiry/audience verify (none of which touch the
database, so an unauthenticated caller cannot make this module read the control plane) and before
the identity and membership lookups; a suspended record raises `TenantSuspendedAtVerification`
there, and an unsuspended one is handed to the membership lookup, whose tenant session routes from
it instead of reading the control plane a second time. A caller that does not ask (the MCP
middleware until #102) gets exactly the old chain, with no suspension check.

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

**One injectable adapter for every control-plane/membership read (#100, prefactor for Spec A1).**
`verify_tenant_token` needs three reads -- a tenant's auth settings, an identity by (issuer,
subject), and a membership's role -- each previously reached by importing a repository class and
a session function (`control_session`/`tenant_session`) at module level and calling it directly.
Those five names are now reached through one object, `ControlPlaneReads`, that `verify_tenant_token`
accepts as an optional `adapter=` keyword and that `app.context_resolution`'s own tenant-record
read (`_read_tenant_record`, the dev-headers path) shares via `default_adapter()` below.
`RepositoryControlPlaneReads` is the only place in this module that still imports the real
repositories and session functions; nothing else here does, and no caller needs to know that.
Tests install one fake of `ControlPlaneReads` (`tests.conftest.FakeControlPlaneReads`) instead of
monkeypatching a repository or a session function on this module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol
from uuid import UUID

import jwt as _pyjwt

from app.config import get_settings
from app.context import RequestContext
from app.db.session import (
    TenantSuspendedError,
    control_session,
    tenant_record_session,
    tenant_session,
)
from app.jwt_verifier import KeySource, TokenVerificationError, verify_token
from app.repositories.control import (
    ControlRepository,
    Identity,
    IdentityRepository,
    TenantAuthSettings,
    TenantAuthSettingsRepository,
)
from app.repositories.memberships import MembershipRepository
from app.tenant_record import TenantRecord

# (issuer) -> the algorithm(s) a token claiming that issuer may be verified with -- the algorithm
# half of the same per-issuer pinning `KeySource` does for the verification key (see
# `app.context_resolution.key_source_for` / `algorithm_source_for`, the two real implementations
# both the HTTP API and the MCP transport use). Never trusts the token's own header for this:
# `verify_token` always passes whatever this resolves to as an explicit allow-list to
# `jwt.decode`, so a token cannot pick its own algorithm, and a token minted under one issuer's
# algorithm/key can never be checked against another issuer's (the algorithm-confusion guard,
# review finding Spec 6).
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
    tenant_record: TenantRecord | None = None
    """The tenant record read during verification -- set exactly when the caller passed
    `read_tenant_record=True` (#104); `None` otherwise."""


class TenantSuspendedAtVerification(TenantSuspendedError):
    """The tenant record read during `verify_tenant_token(..., read_tenant_record=True)` says the
    tenant is suspended (#104). A `TenantSuspendedError` like any other, carrying the issuer the
    token was verified against for the caller's security-event log line."""

    def __init__(self, tenant_id: UUID, *, issuer: str) -> None:
        super().__init__(tenant_id)
        self.issuer = issuer


class ControlPlaneReads(Protocol):
    """The one seam through which `verify_tenant_token` reaches a tenant's auth settings, an
    identity, and a membership role (#100), and through which `app.context_resolution` reads the
    tenant record (#104) -- never a repository class or a session function imported directly.
    Every method takes plain ids/strings and returns a plain value, never a session: an
    implementation owns its own session/transaction, whatever that means for it (a real database,
    an in-memory dict for a test)."""

    async def find_identity_by_issuer_and_subject(
        self, *, issuer: str, subject: str
    ) -> Identity | None: ...

    async def get_tenant_auth_settings(
        self, *, tenant_id: UUID, default_issuer: str | None = None
    ) -> TenantAuthSettings | None: ...

    async def get_membership_role(
        self, *, tenant_id: UUID, identity_id: UUID, tenant_record: TenantRecord | None = None
    ) -> str | None:
        """The identity's role in `tenant_id`, or `None`. `tenant_record`, when given (#104), is
        the record this request already read: the lookup routes by it and reads no control-plane
        row of its own."""
        ...

    async def get_tenant_record(self, *, tenant_id: UUID) -> TenantRecord:
        """`tenant_id`'s control-plane record and settings (#104, `app.tenant_record`), read
        once per request by `app.context_resolution`. Never `None`: a tenant the control plane
        has no row for is the pooled default record."""
        ...


class RepositoryControlPlaneReads:
    """The real, production `ControlPlaneReads` (#100): the only place this module still imports
    `control_session`/`tenant_session` and the three repositories, and calls them directly. What
    was `verify_tenant_token`'s own preliminary-context construction (to look up a membership
    role, ADR-0003 -- issue #91 story 23) is now this class's own internal detail, in
    `get_membership_role` below, rather than the calling function's."""

    async def find_identity_by_issuer_and_subject(
        self, *, issuer: str, subject: str
    ) -> Identity | None:
        async with control_session() as session:
            return await IdentityRepository().find_by_issuer_and_subject(
                session, issuer=issuer, subject=subject
            )

    async def get_tenant_auth_settings(
        self, *, tenant_id: UUID, default_issuer: str | None = None
    ) -> TenantAuthSettings | None:
        async with control_session() as session:
            return await TenantAuthSettingsRepository().get(
                session, tenant_id=tenant_id, default_issuer=default_issuer
            )

    async def get_membership_role(
        self, *, tenant_id: UUID, identity_id: UUID, tenant_record: TenantRecord | None = None
    ) -> str | None:
        # A minimal, role-free context exists only to open the tenant-bound session -- never
        # returned or exposed to a caller (issue #91 story 23: an internal detail of this module).
        # Carrying the request's record, `tenant_session` routes from it without a read (#104).
        preliminary_ctx = RequestContext(
            tenant_id=tenant_id,
            identity_id=identity_id,
            roles=frozenset(),
            tenant_record=tenant_record,
        )
        async with tenant_session(preliminary_ctx) as session:
            return await MembershipRepository().get_role(
                session, preliminary_ctx, identity_id=identity_id
            )

    async def get_tenant_record(self, *, tenant_id: UUID) -> TenantRecord:
        # One transaction on the pooled database, `app.tenant_id` set for it alone: the view and
        # the settings row read together (#104).
        async with tenant_record_session(tenant_id) as session:
            return await ControlRepository().get_tenant_record(session, tenant_id=tenant_id)


# Test-only override installed via `set_default_adapter_for_tests` below -- `None` means "use the
# real, repository-backed adapter." Never read or set anywhere but `default_adapter`/
# `set_default_adapter_for_tests` themselves; production code never touches this name.
_test_default_adapter: ControlPlaneReads | None = None


def set_default_adapter_for_tests(adapter: ControlPlaneReads | None) -> None:
    """Test-only hook (#100), for the no-database suite only: installs `adapter` as what
    `default_adapter()` below returns, for every call to `verify_tenant_token` (and
    `app.context_resolution`'s own tenant-record read) that doesn't pass its own `adapter=`
    explicitly -- the one seam `tests/conftest.py`'s autouse `default_control_plane_reads` fixture
    uses instead of monkeypatching a repository or a session function on this module. Call with
    `None` to restore the real, repository-backed adapter.

    Fails closed outside development: raises `RuntimeError`, installing nothing, unless
    `get_settings().environment` is `dev` or `test` at the moment of the call (code review
    2026-09-26) -- this overrides every auth read of the process."""
    environment = get_settings().environment
    if environment not in ("dev", "test"):
        raise RuntimeError(
            "set_default_adapter_for_tests is a test-only hook and is refused in environment "
            f"{environment!r}; it may only be called with ENVIRONMENT=dev or ENVIRONMENT=test."
        )
    global _test_default_adapter
    _test_default_adapter = adapter


def default_adapter() -> ControlPlaneReads:
    """The adapter `verify_tenant_token` (and `app.context_resolution`'s own tenant-record read)
    use when no explicit `adapter=` is passed: the test override installed via
    `set_default_adapter_for_tests` above, if any, else a fresh `RepositoryControlPlaneReads`."""
    if _test_default_adapter is not None:
        return _test_default_adapter
    return RepositoryControlPlaneReads()


async def verify_tenant_token(
    token: str,
    *,
    tenant_id: UUID,
    key_source: KeySource,
    default_issuer: str | None,
    algorithm_source: AlgorithmSource,
    adapter: ControlPlaneReads | None = None,
    read_tenant_record: bool = False,
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

    `adapter` (#100) is the one seam through which the tenant auth-settings, identity, and
    membership reads below happen; it defaults to `default_adapter()` (the real repositories,
    unless a test has installed an override via `set_default_adapter_for_tests`). A caller never
    needs to pass it explicitly outside a test.

    `read_tenant_record=True` (#104) reads the tenant record between the audience check and the
    identity lookup, raises `TenantSuspendedAtVerification` if it is suspended, routes the
    membership lookup by it, and returns it on `ResolvedIdentity.tenant_record` -- see module
    docstring. Without it, suspension is never checked here.

    Raises `TenantTokenVerificationError` with a single categorized reason on the first check
    that fails; returns the resolved identity and role on success.
    """
    adapter = adapter if adapter is not None else default_adapter()

    if _peek_unverified_issuer(token) == AGENT_IDENTITY_ISSUER:
        # An agent identity's token: tenant-independent issuer (see module docstring) -- skip the
        # tenant auth-settings lookup entirely, since it has nothing to say about this issuer.
        expected_issuer = AGENT_IDENTITY_ISSUER
    else:
        auth_settings = await adapter.get_tenant_auth_settings(
            tenant_id=tenant_id, default_issuer=default_issuer
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

    tenant_record: TenantRecord | None = None
    if read_tenant_record:
        tenant_record = await adapter.get_tenant_record(tenant_id=tenant_id)
        if tenant_record.suspended:
            raise TenantSuspendedAtVerification(tenant_id, issuer=expected_issuer)

    identity = await adapter.find_identity_by_issuer_and_subject(
        issuer=expected_issuer, subject=claims.subject
    )
    if identity is None:
        raise TenantTokenVerificationError(
            VerificationFailureReason.UNKNOWN_IDENTITY, issuer=expected_issuer
        )

    role = await adapter.get_membership_role(
        tenant_id=tenant_id, identity_id=identity.id, tenant_record=tenant_record
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
        tenant_record=tenant_record,
    )
