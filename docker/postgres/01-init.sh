#!/bin/bash
# Committed as non-executable (mode 100644, deliberately — see git history of this file): the
# Postgres entrypoint sources scripts without the execute bit into its own shell instead of
# running them as a subprocess (docker-entrypoint.sh in the official image), so this file needs
# no execute bit and should not carry one.
#
# Runs once, on the first start of the Postgres container, as the cluster's own bootstrap
# superuser ($POSTGRES_USER). That superuser is used here and never again: everything else
# (migrations, seeding, future operator tooling) connects as one of the two roles this script
# creates.
#
#   app_owner — owns every object in the database (schema `public` first, `control` and its
#               objects once a later migration adds that schema). The only role migrations and
#               operator tooling ever connect as. No superuser, no BYPASSRLS: RLS still applies
#               to it wherever a table forces it, exactly like any other non-superuser role.
#   app       — the long-running application role, unchanged in shape: no superuser, no
#               BYPASSRLS (RLS must apply), no CREATEDB/CREATEROLE. Gets a role-level statement
#               timeout and connection limit (see below).
#
# Neither role gets default privileges on future tables: the ALTER DEFAULT PRIVILEGES grant this
# script used to hand `app` on every table that would ever exist is gone. A new table is
# unreadable and unwritable by `app` until the migration that creates it grants access
# explicitly (see migrations/versions/0001_initial.py).
#
# Enabling the pgvector extension stays a superuser-run step here — extension creation needs
# superuser privilege regardless of the role split — so it happens before ownership of the
# schema moves to app_owner.
#
# APP_STATEMENT_TIMEOUT_MS / APP_CONNECTION_LIMIT (Spec 7 / #55): role-level settings, independent
# of the per-transaction statement_timeout the app sets in app/db/session.py's tenant_session().
# They are a deployment-wide ceiling/default that applies even to a connection tenant_session()
# never touched. Defaults here must match app.config.ROLE_STATEMENT_TIMEOUT_MS /
# ROLE_CONNECTION_LIMIT, which the embedded-Postgres integration test asserts against.
set -e
: "${APP_STATEMENT_TIMEOUT_MS:=60000}"
: "${APP_CONNECTION_LIMIT:=50}"
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE ROLE app_owner LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE
        PASSWORD '${APP_OWNER_DB_PASSWORD}';
    GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO app_owner;
    ALTER SCHEMA public OWNER TO app_owner;
    -- Schema-level ownership alone (above) does not let app_owner create a *new* schema:
    -- that needs CREATE on the database itself. The control-plane schema (Spec 1 / #12) is
    -- created by a migration running as app_owner, so this grant is the narrowest one that
    -- makes that possible -- it does not hand app_owner anything beyond "may create schemas".
    GRANT CREATE ON DATABASE ${POSTGRES_DB} TO app_owner;

    CREATE ROLE app LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE
        PASSWORD '${APP_DB_PASSWORD}' CONNECTION LIMIT ${APP_CONNECTION_LIMIT};
    ALTER ROLE app SET statement_timeout = '${APP_STATEMENT_TIMEOUT_MS}ms';
    GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO app;
    GRANT USAGE ON SCHEMA public TO app;
EOSQL
