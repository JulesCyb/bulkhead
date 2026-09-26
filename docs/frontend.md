# Attaching a frontend (Next.js + Vercel AI SDK)

The backend serves the Vercel AI SDK stream format at `POST /v1/t/{tenant_id}/api/chat`
(PydanticAI `VercelAIAdapter`, `sdk_version=6`) — the tenant is named in the URL path
(ADR-0012), never in a header. A Next.js frontend therefore needs no agent code of its own.

Versions move fast — check current docs before starting (`ai`, `@ai-sdk/react`, Next.js).

```bash
npx create-next-app@latest web --ts --app --eslint --tailwind --no-src-dir
cd web && npm i ai @ai-sdk/react
```

Minimal chat (`web/app/page.tsx`, the principle — verify API details against current SDK docs):

```tsx
"use client";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";

export default function Page() {
  const tenantId = process.env.NEXT_PUBLIC_DEV_TENANT_ID!;
  const { messages, sendMessage, status } = useChat({
    transport: new DefaultChatTransport({
      api: `${process.env.NEXT_PUBLIC_API_URL}/v1/t/${tenantId}/api/chat`,
      headers: {                       // dev header — use session/JWT in production
        "X-Identity-Id": process.env.NEXT_PUBLIC_DEV_USER_ID!,
      },
    }),
  });
  // render messages, form -> sendMessage({ text })
}
```

Notes:
- CORS: set `CORS_ORIGINS` in the backend to the frontend URL.
- Tool calls (`search_documents`) arrive as tool parts in the stream — display them so users see what the agent is doing.
- A writing tool (e.g. `rename_document`) can pause mid-stream for the member's approval —
  `useChat`'s deferred-tool-approval flow, not a custom endpoint. The reading/writing split behind
  it lives in `app/agents/run.py`: `chat(adapter)` is the only execution method that binds the
  writing-capable agent, and any writing tool applies the `writing_tool` decorator
  (`app/agents/writing_tools.py`) instead of its own approval bookkeeping — see CLAUDE.md rule 4.
- In production, solve auth in the frontend via the session (cookie/JWT); never ship the dev header.
- The tenant lives only in the URL path (ADR-0012) — a re-login in one tenant's tab can never
  redirect another tenant's tab.
- Alternative without a UI-framework binding: `POST /v1/t/{tenant_id}/agents/assistant/stream`
  (SSE, plain text) or PydanticAI's AG-UI adapter.
