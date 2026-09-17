"""Add refinement fields to agent_runs and create run_stats view

Revision ID: 0003_add_refinement_fields
Revises: e213c35802bb
Create Date: 2026-09-17 00:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0003_add_refinement_fields"
down_revision: str | Sequence[str] | None = "e213c35802bb"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add new fields for review-comment refinement and job dispatch
    op.add_column(
        "agent_runs",
        sa.Column(
            "job_type",
            sa.String(length=32),
            nullable=False,
            server_default="issue",
        ),
    )
    op.add_column(
        "agent_runs",
        sa.Column("review_comment_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "agent_runs",
        sa.Column("parent_run_id", sa.String(length=64), nullable=True),
    )

    # Create run_stats view after table columns exist
    op.execute(
        """
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
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'swe_agent_readonly') THEN
                GRANT SELECT ON run_stats TO swe_agent_readonly;
            END IF;
        END
        $$;
        """
    )


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS run_stats")
    op.drop_column("agent_runs", "parent_run_id")
    op.drop_column("agent_runs", "review_comment_id")
    op.drop_column("agent_runs", "job_type")
