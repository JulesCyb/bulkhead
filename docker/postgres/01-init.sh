#!/bin/bash
# Creates the app role: no superuser, no BYPASSRLS — otherwise Row-Level Security does not apply.
# Runs once on the first start of the Postgres container.
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
    CREATE ROLE app LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE
        PASSWORD '${APP_DB_PASSWORD}' CONNECTION LIMIT ${APP_CONNECTION_LIMIT};
    ALTER ROLE app SET statement_timeout = '${APP_STATEMENT_TIMEOUT_MS}ms';
    GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO app;
    GRANT USAGE ON SCHEMA public TO app;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO app;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO app;
EOSQL
