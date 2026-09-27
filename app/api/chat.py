"""Chat endpoint in the Vercel AI SDK format.

A Next.js frontend using `useChat` (Vercel AI SDK) can talk directly to
POST /v1/t/{tenant_id}/api/chat — PydanticAI translates messages and stream (incl. tool
events). See docs/frontend.md.

Transport only (spec A3 / #93, #108), exactly like the one-shot routes in `app/api/agents.py`:
this route guards the body size, parses the body into the adapter
(`app.agents.run.chat_adapter_from_request`), prepares the run for the adapter's own conversation
(`app.agents.run.prepare_run`), hands the adapter to the prepared run's `chat` method, and maps
the documented errors (`app.agents.run_errors`). Everything the endpoint promises about a chat run
lives in `app/agents/run.py`'s module docstring, "A chat run":

- ADR-0006 / Spec 4 (#33): the only client input trusted is the newest message's
  member-authored content; the history is the server-held one (`ConversationsRepository`), never
  the body's -- `VercelAIAdapter.dispatch_request()` is not used, because its own `messages`
  feed every turn of the body to the agent.
- ADR-0007 (#40): a resumed request's approve/refuse decisions are resolved before the run, and
  only this surface's run binds the writing-capable agent.
- ADR-0006 / Spec 4 (#34): the run's new messages are persisted on completion, decoupled from
  whether the client finishes reading the stream, and the run's conversation id (what tracing
  reports) is tenant-scoped.

Error mapping: a malformed body is a 422 (mirroring `VercelAIAdapter.dispatch_request()`'s own
handling, since this route does not call it); a failure the run table maps (`MAPPED_RUN_ERRORS`,
e.g. an unroutable residency -> 503 `content_routing_unavailable`) before the stream starts is
that table's HTTP error. Once streaming, a run limit or deadline ends the stream with one Vercel
AI SDK `error` chunk instead (the run module's doing).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response
from pydantic import ValidationError

from app.agents.run import chat_adapter_from_request, prepare_run
from app.agents.run_errors import MAPPED_RUN_ERRORS, run_error_http_exception
from app.deps import Context
from app.request_limit import RequestLimit

router = APIRouter(prefix="/api", tags=["chat"])

# Chat histories are legitimately larger than a single prompt, but not unbounded —
# mirrors the 20k prompt cap on /v1/t/{tenant_id}/agents/assistant/*.
MAX_BODY_BYTES = 200_000


@router.post("/chat")
async def chat(request: Request, ctx: Context, _limit: RequestLimit) -> Response:
    if int(request.headers.get("content-length") or 0) > MAX_BODY_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Request body too large")
    # No suspension check here (#106, ADR-0010): `ctx` already carries the record context
    # resolution read and refused a suspended tenant on -- see `app/db/session.py`'s module
    # docstring for the three refusal points that cover every path, this one included.
    try:
        adapter = await chat_adapter_from_request(request)
    except ValidationError as exc:
        try:
            content = exc.json()
        except ValueError:
            content = exc.json(include_input=False)
        return Response(
            content=content,
            media_type="application/json",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )
    try:
        prepared = await prepare_run(ctx, conversation_id=adapter.conversation_id)
        return await prepared.chat(adapter)
    except MAPPED_RUN_ERRORS as exc:
        raise run_error_http_exception(exc) from exc
