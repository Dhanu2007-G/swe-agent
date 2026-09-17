-- infra/postgres/init.sql
-- Runs once on first container start. Sets up roles and permissions.

-- Create a read-only role for dashboards/analytics
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'swe_agent_readonly') THEN
        CREATE ROLE swe_agent_readonly;
    END IF;
END
$$;

-- Grant read-only access to the schema
GRANT CONNECT ON DATABASE swe_agent TO swe_agent_readonly;
GRANT USAGE ON SCHEMA public TO swe_agent_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO swe_agent_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO swe_agent_readonly;

-- Performance: ensure pg_stat_statements is available for query analysis
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Note: Views and app tables are managed via Alembic migrations.

