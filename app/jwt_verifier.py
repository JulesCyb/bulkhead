"""Isolated JWT verification (ADR-0003, ADR-0012, Spec 2 / #24): pure cryptographic checks --
signature, issuer, expiry -- with no database access and no network call of its own.

Key source seam: this module never fetches a verification key itself (no JWKS HTTP call, no
file read). It takes a `KeySource` callable -- `(issuer, kid) -> key` -- so production can wire
up a real key source (a static per-deployment key today; a JWKS client later, see
`app/deps.py::get_key_source`) while tests inject an in-memory one and mint their own tokens.
Nothing here ever reaches the network, which is what keeps this component testable without a
database or an identity provider.

Audience is deliberately NOT checked here: the token's audience must match the URL path's
tenant id (ADR-0012), and this module has no notion of "the current request's path" -- checking
it here would let a caller smuggle a trusted-sounding claim past a component that cannot see the
one thing it needs to be checked against. `app/deps.py` reads `VerifiedClaims.audience` and
compares it itself.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jwt
from jwt import PyJWTError

# (issuer, kid) -> the key to verify the signature with. `kid` is the token header's key id, or
# None when the token carries none. Raising out of a KeySource (e.g. "no key for this issuer")
# is treated by `verify_token` exactly like any other verification failure.
KeySource = Callable[[str, str | None], Any]


# Algorithm-confusion guard, shared by `Settings.jwt_algorithm` and `Settings.agent_token_
# algorithm` (relocated here by spec A4 / #94, #112: this is the pure JWT module, the natural
# home for "which algorithms this codebase ever signs or verifies with" -- `app/config.py`'s
# validators import these to check a deployment's configured algorithms before this module ever
# sees a token). Every algorithm PyJWT's `cryptography` backend supports for signing --
# deliberately never includes "none" or an empty string, so `Settings` construction itself is the
# first place a downgrade-to-unsigned configuration is refused, before `verify_token` (which also
# never accepts "none" -- it always passes an explicit `algorithms` allow-list to `jwt.decode`)
# ever sees a token.
SUPPORTED_JWT_ALGORITHMS = frozenset(
    {
        "HS256",
        "HS384",
        "HS512",
        "RS256",
        "RS384",
        "RS512",
        "ES256",
        "ES384",
        "ES512",
        "PS256",
        "PS384",
        "PS512",
    }
)
HS_ALGORITHMS = frozenset({"HS256", "HS384", "HS512"})
# NIST SP 800-107 / RFC 2104: an HMAC key shorter than its hash's output size is weaker than the
# hash offers -- 32 bytes is the floor for every HS* algorithm above (HS256's own digest size),
# so one constant covers all three rather than sizing per algorithm.
MIN_HS_SECRET_BYTES = 32


class TokenVerificationError(Exception):
    """Every reason a bearer token fails verification: bad signature, expired, wrong issuer,
    malformed, or missing claims. Callers map this to 401 Unauthorized. The message is for logs
    only -- it never repeats the raw token, and callers must not echo `str(exc)` back to the
    client (the response body must stay generic, per ADR-0012's ASGI-seam acceptance criteria)."""


# The four claims verify_token always requires/returns, plus `iat` -- anything else in a
# decoded token's claims is an "extra" claim (see VerifiedClaims.extra below).
_STANDARD_CLAIMS = frozenset({"iss", "sub", "aud", "exp", "iat"})


@dataclass(frozen=True, slots=True)
class VerifiedClaims:
    issuer: str
    subject: str
    audience: str
    # Any claim beyond iss/sub/aud/exp/iat -- e.g. `mint_token`'s `extra_claims` (Spec 6 / gap
    # fix: the credential's public id, so a caller can name it as the request context's means).
    # Never trusted for anything security-relevant on its own; the signature/issuer/audience
    # checks above already ran before this is populated.
    extra: dict[str, str] = field(default_factory=dict)


def verify_token(
    token: str,
    *,
    key_source: KeySource,
    expected_issuer: str,
    algorithms: tuple[str, ...] = ("RS256",),
) -> VerifiedClaims:
    """Verify signature, issuer, and expiry; return the subject and (unverified-against-anything
    -- see module docstring) audience for the caller to check itself.

    Raises `TokenVerificationError` for every failure: malformed token, no key available, bad
    signature, expired, wrong issuer, or a missing `sub`/`aud` claim.
    """
    try:
        header = jwt.get_unverified_header(token)
    except PyJWTError as exc:
        raise TokenVerificationError("malformed token header") from exc

    try:
        key = key_source(expected_issuer, header.get("kid"))
    except Exception as exc:
        raise TokenVerificationError("no verification key for issuer") from exc

    try:
        claims = jwt.decode(
            token,
            key=key,
            algorithms=list(algorithms),
            issuer=expected_issuer,
            options={"require": ["exp", "iss", "sub", "aud"], "verify_aud": False},
        )
    except PyJWTError as exc:
        raise TokenVerificationError("token failed verification") from exc

    subject = claims.get("sub")
    audience = claims.get("aud")
    if not isinstance(subject, str) or not subject:
        raise TokenVerificationError("token missing a usable sub claim")
    if not isinstance(audience, str) or not audience:
        raise TokenVerificationError("token missing a usable aud claim")

    extra = {
        key: value
        for key, value in claims.items()
        if key not in _STANDARD_CLAIMS and isinstance(value, str)
    }
    return VerifiedClaims(issuer=expected_issuer, subject=subject, audience=audience, extra=extra)


def mint_token(
    *,
    subject: str,
    issuer: str,
    audience: str,
    signing_key: str,
    algorithm: str = "RS256",
    ttl_seconds: int,
    extra_claims: dict[str, str] | None = None,
) -> str:
    """Mint a short-lived, signed token (Spec 6 / #47) -- the counterpart to `verify_token`
    above, used only for tokens this application mints itself (today: an agent identity
    exchanging its own credential, `app/agent_credential_exchange.py`). Never used for a
    customer-owned identity provider's own tokens, which this module only ever verifies.

    `signing_key`/`algorithm` must be whatever `verify_token`'s own `key_source` later checks the
    token against -- for the symmetric algorithm this starter defaults to (HS256), that is
    literally the same secret value; no key/algorithm negotiation happens here.

    Carries exactly the four claims `verify_token` requires (`iss`, `sub`, `aud`, `exp`) plus
    `iat` -- nothing else, unless `extra_claims` names more (Spec 6 / gap fix: the credential's
    public id, so a caller can name it as the request context's means without a second lookup at
    verification time).
    """
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": issuer,
        "sub": subject,
        "aud": audience,
        "iat": now,
        "exp": now + ttl_seconds,
    }
    if extra_claims:
        claims.update(extra_claims)
    return jwt.encode(claims, signing_key, algorithm=algorithm)
