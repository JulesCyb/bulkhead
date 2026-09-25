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
#               timeout so one tenant's runaway query cannot hold a shared connection forever.
#
# Neither role gets default privileges on future tables: the ALTER DEFAULT PRIVILEGES grant this
# script used to hand `app` on every table that would ever exist is gone. A new table is
# unreadable and unwritable by `app` until the migration that creates it grants access
# explicitly (see migrations/versions/0001_initial.py).
#
# Enabling the pgvector extension stays a superuser-run step here — extension creation needs
# superuser privilege regardless of the role split — so it happens before ownership of the
# schema moves to app_owner.
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE ROLE app_owner LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE
        PASSWORD '${APP_OWNER_DB_PASSWORD}';
    GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO app_owner;
    ALTER SCHEMA public OWNER TO app_owner;

    CREATE ROLE app LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE PASSWORD '${APP_DB_PASSWORD}';
    GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO app;
    GRANT USAGE ON SCHEMA public TO app;
    ALTER ROLE app SET statement_timeout = '30s';
EOSQL
