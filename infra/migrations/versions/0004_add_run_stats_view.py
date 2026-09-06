"""Add run_stats view

Revision ID: 0004_add_run_stats
Revises: e213c35802bb
Create Date: 2026-09-06 23:50:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0004_add_run_stats'
down_revision: Union[str, Sequence[str], None] = 'e213c35802bb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
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
    """)

    op.execute("GRANT SELECT ON run_stats TO swe_agent_readonly;")


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS run_stats;")
