# Attaching a frontend (Next.js + Vercel AI SDK)

The backend serves the Vercel AI SDK stream format at `POST /api/chat` (PydanticAI
`VercelAIAdapter`, `sdk_version=6`). A Next.js frontend therefore needs no agent code of its own.

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
  const { messages, sendMessage, status } = useChat({
    transport: new DefaultChatTransport({
      api: process.env.NEXT_PUBLIC_API_URL + "/api/chat",
      headers: {                       // dev headers — use session/JWT in production
        "X-Tenant-Id": process.env.NEXT_PUBLIC_DEV_TENANT_ID!,
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
- In production, solve auth in the frontend via the session (cookie/JWT); never ship the dev headers.
- Alternative without a UI-framework binding: `POST /agents/assistant/stream` (SSE, plain text) or PydanticAI's AG-UI adapter.
