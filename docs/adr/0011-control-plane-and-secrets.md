# ADR-0011: Control-plane facts in their own schema, tenant secrets as files, and a real owner role

- **Status:** proposed
- **Date:** 2026-09-12
- **Deciders:** JulesCyb
- **Skill version:** ai-app-blueprints v2.0.0 (research 2026-08)

## Context

ADR-0002, 0003, 0005, 0008, 0009, and 0010 rely on "the control plane" and "a secret store"
without either existing. Verified today: the `app` role holds table-level `UPDATE` on
`tenants`, so under a tenant's own context it may change every column of that row, including
any future `isolation_tier` or `database_alias`; the "owner role" for migrations and seed is the
cluster superuser `postgres`, which bypasses RLS entirely; nothing at runtime checks which role
the application connects as; and default privileges make every future table readable by `app`
before its policy exists.

## Options

1. **Everything in the application database** — control-plane columns on `tenants` with
   column-level grants; secrets as encrypted columns with the key in the environment. Secrets
   end up in every `pg_dump`, and the key sits in the same process.
2. **Own schema plus an external secret manager** — a `control` schema written only by the
   owner role; secrets in Vault or a cloud secret manager, referenced by alias. Best separation,
   one more service to run.
3. **Own schema, secrets as files** — facts as in option 2; secrets never in a database but as
   files per alias under `/run/secrets`, delivered by the deployment (compose secrets, SOPS in
   git, Kubernetes secrets) and read by the application; rotation is a redeploy.

## Decision

We choose **option 3** now and option 2 as the growth step.

- **Roles.** `01-init.sh` creates two roles besides the bootstrap superuser: `app_owner`
  (`NOSUPERUSER NOBYPASSRLS`, owns every object, runs migrations and the operator tool) and
  `app` (unchanged). The superuser is used only for `CREATE EXTENSION` at first start and never
  appears in any file the application or CI reads. The `ALTER DEFAULT PRIVILEGES` line is
  removed; every migration grants explicitly.
- **Control plane.** Schema `control` owned by `app_owner`: tenants' operator-owned facts
  (isolation tier, residency, database alias, suspension state, model allow-list selection),
  identities, and later standing grants. `app` gets `SELECT` through views that expose only what
  a request needs, and `UPDATE` on nothing. Tenant-editable settings stay in `public.tenants`,
  Pydantic-validated, and cannot name a database, a residency, or a model outside the
  allow-list.
- **Tenant secrets.** Files per alias under `/run/secrets`, loaded through pydantic-settings'
  `secrets_dir`; the control plane stores only the alias. Provisioning (ADR-0010) writes the
  file into the deployment, rotation replaces it and notifies the tenant's admins. Agent
  identity credentials (ADR-0005) are not tenant secrets: they are issued to the tenant and
  stored as hashes only.
- **Runtime guard.** At startup the application refuses to run if its role is a superuser or
  has `BYPASSRLS`, or if any table in `public` lacks `FORCE ROW LEVEL SECURITY`; the readiness
  probe repeats the check.

## Consequences

- Positive: a tenant's request cannot move its own tenant between databases or residencies;
  secrets never appear in database backups; migrations no longer bypass RLS by accident;
  misconfigured deployments fail at start instead of leaking.
- Negative / costs: two roles and a schema to explain in the template; provisioning writes to
  the deployment, not only to the database; beyond a few dozen tenants the file-per-alias model
  becomes unwieldy (the trigger for option 2).
- What becomes harder later: none of substance; option 2 replaces the file loader with a
  client and keeps the aliases.

## Revisit when …

- the number of tenant secrets makes files unwieldy, or a customer requires rotation without a
  deploy (then option 2)
- a customer requires keys they own (customer-managed keys; then option 2 with a KMS)
