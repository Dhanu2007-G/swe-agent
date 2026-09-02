from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_repo_cache() -> None:
    from src.tools import _repo_cache

    _repo_cache.invalidate_cache()
    _repo_cache._clone_locks.clear()
    yield
    _repo_cache.invalidate_cache()
    _repo_cache._clone_locks.clear()


class TestGetLocalRepoPath:
    @pytest.mark.asyncio
    async def test_returns_cached_path_without_cloning(self) -> None:
        from src.tools import _repo_cache

        _repo_cache._repo_cache["owner/repo"] = "cached-path"

        with patch("src.tools._repo_cache._clone_fresh", new_callable=AsyncMock) as clone:
            path = await _repo_cache.get_local_repo_path("owner/repo")

        assert path == "cached-path"
        clone.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_clones_and_caches_when_repo_is_missing(self) -> None:
        from src.tools import _repo_cache

        with patch(
            "src.tools._repo_cache._clone_fresh",
            new_callable=AsyncMock,
            return_value="fresh-path",
        ):
            path = await _repo_cache.get_local_repo_path("owner/repo")

        assert path == "fresh-path"
        assert _repo_cache._repo_cache["owner/repo"] == "fresh-path"

    @pytest.mark.asyncio
    async def test_double_checks_cache_after_lock_acquisition(self) -> None:
        from src.tools import _repo_cache

        class CachingLock:
            async def __aenter__(self) -> None:
                _repo_cache._repo_cache["owner/repo"] = "shared-path"

            async def __aexit__(self, *_: object) -> None:
                return None

        _repo_cache._clone_locks["owner/repo"] = CachingLock()

        with patch("src.tools._repo_cache._clone_fresh", new_callable=AsyncMock) as clone:
            path = await _repo_cache.get_local_repo_path("owner/repo")

        assert path == "shared-path"
        clone.assert_not_awaited()


class TestCloneFresh:
    @pytest.mark.asyncio
    async def test_clones_repo_with_authenticated_kwargs(self) -> None:
        from src.tools import _repo_cache

        temp_path = str(Path("test_out") / "repo-cache" / "owner-repo")
        loop = MagicMock()

        async def run_in_executor(_: object, fn: object) -> None:
            fn()

        loop.run_in_executor = AsyncMock(side_effect=run_in_executor)

        with (
            patch("src.tools._repo_cache.tempfile.mkdtemp", return_value=temp_path),
            patch(
                "src.tools._repo_cache._build_authenticated_clone_kwargs",
                return_value={
                    "url": "https://github.com/owner/repo.git",
                    "multi_options": ["-c", "header"],
                },
            ),
            patch("src.tools._repo_cache.asyncio.get_running_loop", return_value=loop),
            patch("src.tools._repo_cache.git.Repo.clone_from") as mock_clone,
        ):
            cloned = await _repo_cache._clone_fresh("owner/repo")

        assert cloned == temp_path
        mock_clone.assert_called_once_with(
            "https://github.com/owner/repo.git",
            temp_path,
            depth=1,
            multi_options=["-c", "header"],
        )


class TestInvalidateCache:
    def test_invalidates_single_repo_and_entire_cache(self) -> None:
        from src.tools import _repo_cache

        _repo_cache._repo_cache["owner/repo"] = "path-1"
        _repo_cache._repo_cache["owner/other"] = "path-2"

        with patch("shutil.rmtree") as mock_rmtree:
            _repo_cache.invalidate_cache("owner/repo")
            _repo_cache.invalidate_cache()

        assert "owner/repo" not in _repo_cache._repo_cache
        mock_rmtree.assert_any_call("path-1", ignore_errors=True)
        mock_rmtree.assert_any_call("path-2", ignore_errors=True)

    def test_build_authenticated_clone_kwargs(self) -> None:
        from src.tools._repo_cache import _build_authenticated_clone_kwargs
        from types import SimpleNamespace

        settings = SimpleNamespace(github_token_value="ghp_test123")
        res = _build_authenticated_clone_kwargs("owner/repo", settings)
        assert res["url"] == "https://ghp_test123@github.com/owner/repo.git"
        assert res["depth"] == 1
