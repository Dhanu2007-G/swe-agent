"""Create agent_runs table

Revision ID: 0001
Revises: 
Create Date: 2026-01-01 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("repo_full_name", sa.String(length=255), nullable=False),
        sa.Column("issue_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("pr_url", sa.String(length=500), nullable=True),
        sa.Column("pr_number", sa.Integer(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("state_snapshot", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    # Unique index on run_id (the primary lookup key)
    op.create_index(
        op.f("ix_agent_runs_run_id"),
        "agent_runs",
        ["run_id"],
        unique=True,
    )
    # Composite index for looking up runs by repo + issue
    op.create_index(
        "ix_agent_runs_repo_issue",
        "agent_runs",
        ["repo_full_name", "issue_number"],
        unique=False,
    )
    # Index for filtering by status (e.g., find all "running" jobs)
    op.create_index(
        "ix_agent_runs_status",
        "agent_runs",
        ["status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_agent_runs_status", table_name="agent_runs")
    op.drop_index("ix_agent_runs_repo_issue", table_name="agent_runs")
    op.drop_index(op.f("ix_agent_runs_run_id"), table_name="agent_runs")
    op.drop_table("agent_runs")
