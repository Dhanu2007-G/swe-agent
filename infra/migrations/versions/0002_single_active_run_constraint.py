"""Enforce a single active run per repo/issue in Postgres.

Revision ID: 0002_single_active
Revises: 0001_initial_schema
Create Date: 2026-03-28 00:00:00.000000
"""
from __future__ import annotations

from alembic import op

revision = "0002_single_active"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        WITH ranked_active_runs AS (
            SELECT
                id,
                ROW_NUMBER() OVER (
                    PARTITION BY repo_full_name, issue_number
                    ORDER BY
                        CASE WHEN status = 'running' THEN 0 ELSE 1 END,
                        created_at DESC,
                        id DESC
                ) AS row_rank
            FROM agent_runs
            WHERE status IN ('pending', 'running')
        )
        UPDATE agent_runs
        SET
            status = 'failed',
            completed_at = COALESCE(completed_at, now()),
            failure_reason = COALESCE(
                failure_reason,
                'Superseded during migration enforcing single active run'
            )
        WHERE id IN (
            SELECT id
            FROM ranked_active_runs
            WHERE row_rank > 1
        )
        """
    )

    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_runs_single_active_issue
        ON agent_runs (repo_full_name, issue_number)
        WHERE status IN ('pending', 'running')
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_agent_runs_single_active_issue")
