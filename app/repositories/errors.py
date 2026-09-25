"""Shared exception family for one repository rule: **fail closed, never leak existence across
tenants.**

Finding from the 2026-09-25 review of #46/#38: a repository that refuses because a target id
belongs to another tenant, doesn't exist at all, or exists here but fails some other in-tenant
invariant (e.g. a membership that isn't the role a caller needed) must answer the *same* way for
all of those reasons -- a plain "not found", never a message that lets a caller tell them apart.
`app.repositories.agent_credentials.UnknownAgentIdentity` got this right the first time (and its
own 404, `app.main.handle_not_found_in_tenant`); `app.repositories.standing_grants
.NotAnAgentMembership` shared the exact same reasoning in its docstring but had no handler at all,
so it fell through to the generic 500 -- an information leak in its own right (500 vs 404 tells a
caller "this id exists somewhere" in a way a 404 doesn't) as well as a broken caller experience.

**Subclass `NotFoundInTenant` instead of adding a new `ValueError` + a new per-exception handler
in `app/main.py`.** One handler (`handle_not_found_in_tenant`) is registered once, on this base
class, and FastAPI/Starlette dispatch exceptions by walking the MRO -- so every subclass reaches
it automatically. Set `public_message` on your subclass to the one fixed sentence every caller
sees, regardless of which of the "doesn't exist" / "other tenant" / "exists but wrong kind" cases
actually happened; `str(exc)` (the constructor argument), which may name real ids, is logged for
operators and never put in the response.
"""

from __future__ import annotations


class NotFoundInTenant(ValueError):
    """Base for every repository condition meaning "nothing in the caller's own tenant matches
    this id" -- raise a subclass, never this class directly, and set `public_message` on it.

    Fail closed: a repository raises this the moment it cannot positively confirm the target is
    both real and in-tenant, and does not attempt to be more specific for the caller's benefit --
    more specific is exactly the information this rule exists to withhold. See the module
    docstring for the full reasoning and `app.main.handle_not_found_in_tenant` for the one place
    this becomes an HTTP response.
    """

    public_message: str = "Not found in this tenant."
