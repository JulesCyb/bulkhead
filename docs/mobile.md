# Attaching a mobile app (Android/iOS)

The app is another client of the same API — agent logic, tools, RLS, and model access stay
unchanged. Decision and options: the `ai-app-blueprints` skill, section "Mobile clients"
and ADR template `adr-0005-mobile.md`.

## Paths (check in this order)

1. **PWA / packaged web app** (Capacitor, Trusted Web Activity): the Next.js frontend as an
   installable app — the cheapest test, often enough for internal tools.
2. **Expo / React Native**: same language as the web frontend; the Vercel AI SDK also runs in
   React Native and talks to `POST /v1/t/{tenant_id}/api/chat`. Monorepo with a shared package
   (types, hooks, API client generated from the OpenAPI schema at `/openapi.json`).
3. **Native (Kotlin/Compose)**: only with a reason — deep OS integration, a native team.

## Backend duties before the first app release

- Set `AUTH_MODE=jwt` (implemented in `app/deps.py`, issue #24): OIDC + PKCE with an identity
  provider, short-lived access tokens per tenant (the token's audience must name the tenant in
  the URL path — ADR-0012), refresh tokens, revocation. The dev headers are off-limits on a
  device.
- Freeze API contracts: `/v1/t/{tenant_id}/` prefix (ADR-0012 — the tenant lives in the path, a
  single `X-Identity-Id`/token identifies the caller, never a tenant header), no breaking
  changes, `/openapi.json` as the contract, generate a client (e.g. `openapi-typescript`), keep
  a deprecation window.
- Long agent runs as jobs: `POST /v1/jobs` → status endpoint or push → result in the DB
  (a table with `tenant_id` + RLS). SSE streaming only for chat.
- Uploads via signed URLs to object storage (tenant prefix), processing server-side.
- Push (FCM/APNs): device tokens per user with `tenant_id`; no content in payloads.
- Per-user rate limits; Play Integrity / App Attest only when needed.

## What is not needed

AI on the device. Models stay behind the API; on-device models are at most a later addition
for small offline tasks.
