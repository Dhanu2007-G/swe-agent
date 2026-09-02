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

-- Useful indexes for common dashboard queries
-- (applied after table creation by alembic, added here as reference)

-- Useful view for run analytics
CREATE OR REPLACE VIEW run_stats AS
SELECT
    date_trunc('day', created_at) AS day,
    repo_full_name,
    status,
    COUNT(*) AS total_runs,
    AVG(retry_count) AS avg_retries,
    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) AS avg_duration_seconds,
    SUM(CASE WHEN status = 'succeeded' THEN 1 ELSE 0 END)::float /
        NULLIF(COUNT(*), 0) * 100 AS solve_rate_pct
FROM agent_runs
WHERE completed_at IS NOT NULL
GROUP BY 1, 2, 3
ORDER BY 1 DESC, 4 DESC;

GRANT SELECT ON run_stats TO swe_agent_readonly;
