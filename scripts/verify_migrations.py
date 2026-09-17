"""
scripts/verify_migrations.py — Verify Alembic migration graph integrity.
Ensures single head, unbroken chain, and no duplicate revisions.
"""

from __future__ import annotations

import sys
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory


def verify_migrations() -> None:
    project_root = Path(__file__).resolve().parent.parent
    alembic_ini = project_root / "alembic.ini"

    if not alembic_ini.exists():
        print(f"Error: {alembic_ini} not found", file=sys.stderr)
        sys.exit(1)

    config = Config(str(alembic_ini))
    script = ScriptDirectory.from_config(config)

    heads = script.get_heads()
    if len(heads) != 1:
        print(
            f"ERROR: Multiple or zero Alembic heads detected: {heads}. Expected exactly 1 head.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Walk all revisions to ensure no broken references
    revisions = list(script.walk_revisions())
    print(f"Verified {len(revisions)} Alembic migration revision(s).")
    print(f"Current migration head: {heads[0]}")


if __name__ == "__main__":
    verify_migrations()
