# Connecting to the tools server (ADR-0005)

The MCP server (`app/mcp/server.py`) hands the same tool functions (`search_documents`,
`list_memberships`) to Claude Code, Claude Desktop, or any other MCP client. Which identity a
connection runs as, and whether a token is required at all, depends on the transport
(`MCP_TRANSPORT`, `.env.example`):

| Transport | Who it's for | Identity comes from | Token required |
|---|---|---|---|
| `stdio` (default) | local development only | the process's own `MCP_TENANT_ID`/`MCP_IDENTITY_ID` | no |
| `streamable-http` | production — the only path outside local development | the connection's own bearer token, verified per connection (`app.token_verifier.verify_tenant_token`) | **yes** |

`app.mcp.server.check_mcp_mode` enforces this at startup, fail-closed: `stdio`'s process-wide
identity refuses to start outside `ENVIRONMENT=dev`/`test` (mirroring how `AUTH_MODE=dev-headers`
is guarded), and `streamable-http` refuses to start at all unless `JWT_VERIFICATION_KEY` is
configured. There is no way to run a networked, unauthenticated MCP endpoint.

## As a member: connecting your own identity (stdio, local development)

Use this path while developing against your own tenant, running the API on your own machine.

1. Create a tenant and your own identity if you have not already:
   `uv run python scripts/operator.py create "My Tenant" --residency eu --admin-email me@example.com`
   — this prints the tenant id and your identity id.
2. Copy `.mcp.json.example` to `.mcp.json` (Claude Code) or your MCP client's own config file
   location (Claude Desktop), keeping only the `ai-app-tools` (stdio) entry.
3. Fill in the placeholders:
   - `args`: replace `/absolute/path/to/ai-app-starter` with this repo's absolute path on disk —
     `uv run --directory <path> …` tells `uv` where the project lives explicitly, so the
     server's `.env` resolves correctly no matter which directory the MCP client happens to
     launch the command from (previously an assumption the client's own working directory
     matched the repo).
   - `env.MCP_TENANT_ID` / `env.MCP_IDENTITY_ID`: the two ids from step 1.
4. Restart Claude Code / Claude Desktop. Tool calls now run as you, in your tenant: `search_documents`
   returns what your own membership can see, and (if your membership's role is `admin`)
   `list_memberships` works too — exactly the role check `ctx.require_role(...)` already applies
   to the HTTP API's equivalent routes.

`stdio` refuses to start once `ENVIRONMENT` is anything but `dev`/`test` — this path is not meant
to reach further than your own machine.

## As a member: connecting over the network (streamable-http, production)

Outside local development, the MCP server speaks Streamable HTTP and is mounted at
`/v1/t/{tenant_id}/mcp` (ADR-0012 — the same tenant-in-the-path rule every HTTP route follows).
Every connection needs a bearer token whose audience is that same tenant.

1. Copy the `ai-app-tools-remote` entry from `.mcp.json.example`.
2. Set `url` to `https://<your-deployment-host>/v1/t/<TENANT_ID>/mcp`.
3. Set the `Authorization` header to `Bearer <TOKEN>` — a token minted for your own identity by
   whatever issues your tenant's person tokens (`AUTH_MODE=jwt`, `app/deps.py`).
4. Connect. `MCPTenantAuthMiddleware` verifies the token via the exact module the HTTP API uses
   (`app.token_verifier.verify_tenant_token`) and resolves your context by delegation: you are
   the actor, the assistant's tools are the means (`RequestContext.acting_through("agent",
   "assistant")`) — the same shape a chat message through the web UI produces.

A token naming a different tenant than the one in the URL is refused with 403, matching
ADR-0012's three-way check on the HTTP side; an expired or otherwise invalid token gets 401 —
reconnect with a fresh one. Either rejection is generic (`"Not authorized for this tenant."` /
`"Invalid or expired token"`) and never reveals which part was wrong.

## As a tenant admin: creating an agent identity and its first credential

Use this to give automation you own (a scheduled job, a script) an identity of its own, instead
of borrowing a person's. Every route below requires role `admin` in the tenant
(`ctx.require_role("admin")`, ADR-0004) and lives under `/v1/t/{tenant_id}/`.

1. **Create the agent identity** — `POST /v1/t/{tenant_id}/agent-identities`, body
   `{"name": "nightly sync"}`. Creates an identity of kind `agent` with a membership of role
   `agent` in your tenant (reading tools only — a writing tool needs the standing grant Spec 5
   defines, never the credential itself).
2. **Issue it a credential** — `POST /v1/t/{tenant_id}/agent-identities/{identity_id}/credentials`,
   body `{"name": "nightly sync — prod"}`. The response's `secret` field is shown exactly once,
   here — the application stores only its hash (`AgentCredentialRepository`, SHA-256 of a
   256-bit random token) and can never show it again.
3. **Hand the `public_id` and `secret` to your automation.** It exchanges them for a
   short-lived access token by calling `POST /v1/t/{tenant_id}/agent-tokens` with
   `{"public_id": "...", "secret": "..."}`. The response's `access_token` is what goes into that
   automation's own `Authorization: Bearer` header — the same `ai-app-tools-remote` shape a
   person uses, just with an agent identity's token instead of a person's. A wrong secret, an
   unknown `public_id`, and a revoked credential all fail the exact same way
   (`401 Invalid credential.`) — never distinguishable from one another.
4. **List what you've issued** — `GET /v1/t/{tenant_id}/agent-credentials` — name, creation
   date, last use, and whether it is revoked, for every credential in your tenant. Never a
   secret or its hash.
5. **Revoke a credential** — `POST /v1/t/{tenant_id}/agent-credentials/{credential_id}/revoke`.
   Immediate and idempotent: revoking an already-revoked (or unknown, or cross-tenant)
   credential returns `{"revoked": false}` rather than erroring. A revoked credential's next
   token exchange fails closed — no already-issued token quietly outlives the revocation past
   its own short TTL (`AGENT_TOKEN_TTL_SECONDS`, five minutes by default).

Every tool call an agent identity's token makes is recorded as that agent identity, naming the
credential that authenticated it (`RequestContext.acting_through("credential", public_id)`) — an
auditor can always answer "who, and through what" without reconstructing the connection from raw
logs.

## Errors you can act on

| Response | Meaning | What to do |
|---|---|---|
| `401 Missing or malformed bearer token` | no/garbled `Authorization` header | add `Authorization: Bearer <token>` |
| `401 No token issuer configured` / `401 Invalid or expired token` | the token doesn't verify | reconnect with a fresh token |
| `403 Not authorized for this tenant.` | the token's tenant doesn't match the URL's | use the token minted for this tenant |
| a tool call's own error | server-side exception | logged server-side; the client sees only `"This tool call failed. The error has been logged."` (except a missing-role `PermissionError`, which names the missing role, and a suspended-tenant rejection) — never the raw exception |
