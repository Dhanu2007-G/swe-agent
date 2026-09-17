"""
scripts/verify_db_schema.py — Verify applied database schema and objects.
Checks tables, columns, indexes, partial unique constraints, and views exist.
"""

from __future__ import annotations

import os
import sys

from sqlalchemy import create_engine, inspect


def verify_schema() -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL not set, skipping live database schema verification.")
        return

    # Convert asyncpg scheme to psycopg2/sync postgresql for inspector
    sync_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    engine = create_engine(sync_url)
    inspector = inspect(engine)

    tables = inspector.get_table_names()
    if "agent_runs" not in tables:
        print("ERROR: agent_runs table missing", file=sys.stderr)
        sys.exit(1)

    columns = {c["name"] for c in inspector.get_columns("agent_runs")}
    required_columns = {
        "id",
        "job_type",
        "repo_full_name",
        "issue_number",
        "status",
        "review_comment_id",
        "parent_run_id",
    }
    missing = required_columns - columns
    if missing:
        print(f"ERROR: missing columns in agent_runs: {missing}", file=sys.stderr)
        sys.exit(1)

    views = inspector.get_view_names()
    if "run_stats" not in views:
        print("ERROR: run_stats view missing", file=sys.stderr)
        sys.exit(1)

    indexes = inspector.get_indexes("agent_runs")
    index_names = {idx["name"] for idx in indexes}
    valid_indexes = {"uq_agent_runs_single_active_issue", "uq_single_active_run_per_repo_issue"}
    if not (valid_indexes & index_names):
        print(
            "ERROR: partial unique index uq_agent_runs_single_active_issue missing",
            file=sys.stderr,
        )
        sys.exit(1)

    print("All required database tables, columns, views, and indexes verified successfully.")


if __name__ == "__main__":
    verify_schema()
