"""
src/agent/checkpointing.py — LangGraph Postgres checkpointer setup.
Enables:
  - Resuming a failed run mid-graph
  - Full state snapshot audit trail
  - Human-in-the-loop breakpoints (future)
  - Parallel runs without state collision (thread_id isolation)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.config import get_settings

log = structlog.get_logger(__name__)

_checkpointer: AsyncPostgresSaver | None = None


async def get_checkpointer() -> AsyncPostgresSaver:
    """
    Return a shared checkpointer instance.
    LangGraph's AsyncPostgresSaver manages its own connection pool.
    """
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    settings = get_settings()

    # LangGraph expects a sync psycopg connection string (not asyncpg)
    pg_url = str(settings.database_url).replace("postgresql+asyncpg://", "postgresql://")

    _checkpointer = AsyncPostgresSaver.from_conn_string(pg_url)  # type: ignore[assignment]
    await getattr(_checkpointer, "setup")()  # creates langgraph checkpoint tables

    log.info("checkpointer.initialized")
    return _checkpointer  # type: ignore[return-value]


@asynccontextmanager
async def checkpointed_run(run_id: str) -> AsyncIterator[AsyncPostgresSaver]:
    """
    Context manager for a single checkpointed run.

    Usage:
        async with checkpointed_run(run_id) as cp:
            graph = await get_compiled_graph(checkpointer=cp)
            await graph.ainvoke(state, config={"configurable": {"thread_id": run_id}})

    State is snapshotted after EVERY node. If the process crashes mid-run,
    resume by calling ainvoke again with the same thread_id — LangGraph
    replays from the last checkpoint.
    """
    cp = await get_checkpointer()
    log.info("checkpointer.run_start", run_id=run_id)
    try:
        yield cp
        log.info("checkpointer.run_complete", run_id=run_id)
    except Exception as e:
        log.error("checkpointer.run_error", run_id=run_id, error=str(e))
        raise


async def get_run_state(run_id: str) -> dict | None:
    """
    Retrieve the last saved state for a run_id.
    Useful for debugging failed runs or inspecting partial progress.
    """
    cp = await get_checkpointer()
    config = {"configurable": {"thread_id": run_id}}
    checkpoint = await cp.aget(config)  # type: ignore[arg-type]
    if checkpoint is None:
        return None
    return checkpoint.get("channel_values", {})


async def list_checkpointed_runs(limit: int = 20) -> list[dict]:
    """List recent runs that have checkpoints (not just DB records)."""
    cp = await get_checkpointer()
    runs = []
    async for checkpoint_tuple in cp.alist({}):
        metadata = checkpoint_tuple.metadata or {}
        config = checkpoint_tuple.config or {}
        runs.append(
            {
                "thread_id": config.get("configurable", {}).get("thread_id"),
                "step": checkpoint_tuple.checkpoint.get("id"),
                "created_at": metadata.get("created_at"),
            }
        )
        if len(runs) >= limit:
            break
    return runs
