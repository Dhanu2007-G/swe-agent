"""
src/tools/_repo_cache.py — In-process LRU cache for cloned repos.
Prevents redundant git clones within the same worker process.
"""

from __future__ import annotations

import asyncio
import tempfile
from typing import Any

import git
import structlog

from src.config import get_settings

log = structlog.get_logger(__name__)

# Map of repo_full_name -> local temp dir path
_repo_cache: dict[str, str] = {}
_clone_locks: dict[str, asyncio.Lock] = {}


async def get_local_repo_path(repo_full_name: str) -> str:
    """
    Return path to a locally cloned repo.
    Clones once per worker process, then reuses.
    Uses per-repo locks to prevent concurrent duplicate clones.
    """
    if repo_full_name in _repo_cache:
        return _repo_cache[repo_full_name]

    if repo_full_name not in _clone_locks:
        _clone_locks[repo_full_name] = asyncio.Lock()

    async with _clone_locks[repo_full_name]:
        # Double-check after acquiring lock
        if repo_full_name in _repo_cache:
            return _repo_cache[repo_full_name]

        path = await _clone_fresh(repo_full_name)
        _repo_cache[repo_full_name] = path
        return path


def _build_authenticated_clone_kwargs(repo_full_name: str, settings: Any) -> dict[str, Any]:
    """Helper to build clone kwargs with authentication."""
    token = getattr(settings, "github_token_value", getattr(settings, "github_token", ""))
    url = f"https://{token}@github.com/{repo_full_name}.git"
    return {"url": url, "depth": 1}


async def _clone_fresh(repo_full_name: str) -> str:
    settings = get_settings()
    tmp = tempfile.mkdtemp(prefix="swe-agent-repo-")
    clone_kwargs = _build_authenticated_clone_kwargs(repo_full_name, settings)
    url = clone_kwargs.pop(
        "url",
        f"https://{getattr(settings, 'github_token_value', '')}@github.com/{repo_full_name}.git",
    )
    clone_kwargs.setdefault("depth", 1)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        lambda: git.Repo.clone_from(url, tmp, **clone_kwargs),
    )
    log.info("repo_cache.cloned", repo=repo_full_name, path=tmp)
    return tmp


def invalidate_cache(repo_full_name: str | None = None) -> None:
    """Call between tests to avoid stale repos."""
    import shutil

    if repo_full_name:
        path = _repo_cache.pop(repo_full_name, None)
        if path:
            shutil.rmtree(path, ignore_errors=True)
    else:
        for path in _repo_cache.values():
            shutil.rmtree(path, ignore_errors=True)
        _repo_cache.clear()
