"""Membership listing: the role-gated example route (ADR-0004, S3-T1 / #26).

`list_memberships` (app/tools/memberships.py) checks the caller's role before doing anything
else; this route calls that same tool, so the enforcement lives in exactly one place and any
future MCP exposure of the same tool inherits it unchanged. A non-admin role never reaches the
repository — it is refused with a 403 from the registered exception handler
(`app.main.handle_permission_error`), never a crash.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.deps import Context
from app.repositories.memberships import MembershipRecord
from app.tools.memberships import list_memberships

router = APIRouter(prefix="/memberships", tags=["memberships"])


class MembershipListResponse(BaseModel):
    memberships: list[MembershipRecord]


@router.get("", response_model=MembershipListResponse)
async def list_tenant_memberships(ctx: Context) -> MembershipListResponse:
    records = await list_memberships(ctx)
    return MembershipListResponse(memberships=records)
