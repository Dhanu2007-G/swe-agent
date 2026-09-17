"""
src/db/database.py + models.py — Async SQLAlchemy setup with Alembic migrations.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from sqlalchemy import (
    DateTime,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.config import get_settings

# ── Engine ────────────────────────────────────────────────────────────────────

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_db() -> None:
    """Initialize DB engine and create tables. Called on app startup."""
    global _engine, _session_factory
    settings = get_settings()

    _engine = create_async_engine(
        str(settings.database_url),
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        echo=settings.database_echo,
        pool_pre_ping=True,  # validate connections before use
        pool_recycle=1800,  # recycle after 30 min
    )

    _session_factory = async_sessionmaker(
        bind=_engine,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    # In production, use Alembic migrations only.
    # Keep create_all() only for test/dev if explicitly configured
    auto_create = getattr(settings, "database_auto_create_tables", True)
    is_prod = getattr(settings, "is_production", False)
    if auto_create and not is_prod:
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncIterator[AsyncSession]:
    """Dependency-injectable async session."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── ORM Models ────────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


class AgentRun(Base):
    """Persistent record of every agent run."""

    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    repo_full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    issue_number: Mapped[int] = mapped_column(Integer, nullable=False)
    job_type: Mapped[str] = mapped_column(String(32), default="issue", nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    pr_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    pr_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    review_comment_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    parent_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    state_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_agent_runs_repo_issue", "repo_full_name", "issue_number"),
        Index("ix_agent_runs_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<AgentRun run_id={self.run_id!r} issue={self.issue_number} status={self.status!r}>"
