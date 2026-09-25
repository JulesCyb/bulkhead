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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jwt
from jwt import PyJWTError

# (issuer, kid) -> the key to verify the signature with. `kid` is the token header's key id, or
# None when the token carries none. Raising out of a KeySource (e.g. "no key for this issuer")
# is treated by `verify_token` exactly like any other verification failure.
KeySource = Callable[[str, str | None], Any]


class TokenVerificationError(Exception):
    """Every reason a bearer token fails verification: bad signature, expired, wrong issuer,
    malformed, or missing claims. Callers map this to 401 Unauthorized. The message is for logs
    only -- it never repeats the raw token, and callers must not echo `str(exc)` back to the
    client (the response body must stay generic, per ADR-0012's ASGI-seam acceptance criteria)."""


@dataclass(frozen=True, slots=True)
class VerifiedClaims:
    issuer: str
    subject: str
    audience: str


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

    return VerifiedClaims(issuer=expected_issuer, subject=subject, audience=audience)
