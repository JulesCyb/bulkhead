"""Agent-credential token exchange (ADR-0005, ADR-0012, Spec 6 / #47): a presented credential
(public id + secret) is exchanged for a short-lived, signed access token scoped to the one
tenant that issued it, naming the agent identity as its subject.

Reuses #45's `AgentCredentialRepository.verify_and_touch` for the credential check -- an unknown
identifier, a wrong secret, and a revoked credential all resolve to `None` there, indistinguishably
-- and #44's signing/verification material (`app/jwt_verifier.py`) for the token itself, so the
same "tenant as audience" shape ADR-0012 already applies to person tokens applies here too, and a
single module (`app/token_verifier.py`) verifies both kinds later.

**Gap fix (Spec 6 / #49):** the minted token's issuer is the agent identity's *own* issuer
(`token_verifier.AGENT_IDENTITY_ISSUER`, the fixed literal `control.create_agent_identity`
synthesizes every agent identity's `issuer` column as) -- never the requesting tenant's own
human-IdP issuer. The two used to be conflated here, which meant a real agent token could never
verify: `verify_tenant_token` resolved the tenant's human issuer and checked the token's signature
against `jwt_verification_key`, while this module signed with the separate
`agent_token_signing_key`. Signing under the agent identity's real issuer lets the shared verifier
recognize it (by peeking `iss`, see `app/token_verifier.py`) and check it against the matching key
instead. The credential's own public id travels along as an extra `cred` claim so a caller
resolving the token later (the MCP transport, #49) can name it as `RequestContext`'s means without
a second lookup.

This module raises exactly one exception, `AgentCredentialExchangeError`, for every way an
exchange can fail -- the caller (`app/api/agent_tokens.py`) maps it to one generic response,
never revealing which part of a bad attempt was wrong. The exception message is for logs only.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from uuid import UUID

from app.config import Settings
from app.context import RequestContext
from app.db.session import control_session, tenant_session
from app.jwt_verifier import mint_token
from app.repositories.agent_credentials import AgentCredentialRepository
from app.repositories.control import IdentityRepository
from app.token_verifier import AGENT_IDENTITY_ISSUER

# A well-known nil UUID, never a real identity: this exchange has no caller identity yet -- it is
# how one is first obtained. It only ever scopes the one tenant-bound transaction that checks the
# presented credential (RLS on agent_credentials filters on tenant_id alone); it is never
# persisted, minted into a token, or returned to anything outside this function.
_NO_IDENTITY_YET = uuid.UUID(int=0)


class AgentCredentialExchangeError(Exception):
    """Raised for every way an exchange can fail: an unknown public id, a wrong secret, a revoked
    credential, or a tenant/identity that cannot be resolved to a usable issuer. Callers map this
    to a single generic response -- see module docstring."""


@dataclass(frozen=True, slots=True)
class IssuedAgentToken:
    access_token: str
    token_type: str
    expires_in: int


async def exchange_agent_credential(
    *, tenant_id: UUID, public_id: str, secret: str, settings: Settings
) -> IssuedAgentToken:
    """Verify a presented credential against `tenant_id` and mint a short-lived access token for
    the agent identity it belongs to.

    A successful call advances the credential's last-used marker (via `verify_and_touch`); a
    failed call -- unknown identifier, wrong secret, or revoked -- never does, and always raises
    `AgentCredentialExchangeError` rather than returning a distinguishable failure value.
    """
    preliminary_ctx = RequestContext(
        tenant_id=tenant_id, identity_id=_NO_IDENTITY_YET, roles=frozenset()
    )
    async with tenant_session(preliminary_ctx) as session:
        verified = await AgentCredentialRepository().verify_and_touch(
            session, preliminary_ctx, public_id=public_id, secret=secret
        )
    if verified is None:
        raise AgentCredentialExchangeError("unknown credential, wrong secret, or revoked")

    async with control_session() as session:
        identity = await IdentityRepository().get_by_id(session, identity_id=verified.identity_id)
    if identity is None:
        # Should not happen in practice (the credential's own FK guarantees the identity exists),
        # but never mint a token for an identity this exchange could not itself resolve.
        raise AgentCredentialExchangeError("credential's identity does not resolve")

    # Gap fix (see module docstring): the agent identity's own issuer, never the tenant's
    # human-IdP one -- `control.create_agent_identity` (migration 0032) is the only writer of
    # `identity.issuer` for an identity of kind agent, and it always synthesizes exactly this
    # value. A mismatch here would mean this credential does not actually belong to an agent
    # identity at all -- fail closed rather than mint a token nothing can later verify correctly.
    if identity.issuer != AGENT_IDENTITY_ISSUER:
        raise AgentCredentialExchangeError("credential's identity is not an agent identity")
    if settings.agent_token_signing_key is None:
        raise AgentCredentialExchangeError("no agent token signing key configured")

    ttl_seconds = settings.agent_token_ttl_seconds
    access_token = mint_token(
        subject=identity.subject,
        issuer=identity.issuer,
        audience=str(tenant_id),
        signing_key=settings.agent_token_signing_key.get_secret_value(),
        # Deliberately agent_token_algorithm, never jwt_algorithm (review finding, Spec 6): the
        # two are independent settings for independent token populations -- see app/config.py's
        # agent_token_algorithm docstring and app.deps.get_key_source / get_algorithm_source,
        # which pin the matching key/algorithm pair per issuer at verification time.
        algorithm=settings.agent_token_algorithm,
        ttl_seconds=ttl_seconds,
        # The credential's own public id, so a caller resolving this token later (the MCP
        # transport, #49) can name it as `RequestContext`'s means without a second lookup.
        extra_claims={"cred": public_id},
    )
    return IssuedAgentToken(access_token=access_token, token_type="bearer", expires_in=ttl_seconds)
