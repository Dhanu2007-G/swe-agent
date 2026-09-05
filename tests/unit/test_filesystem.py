"""
tests/unit/test_filesystem.py — Tests for file loading, BM25 search, and token budget.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.tools.filesystem import (
    LANGUAGE_MAP,
    _extract_symbols,
    _tokenize_for_bm25,
    count_tokens,
)


# ── Language Detection ────────────────────────────────────────────────────────


class TestLanguageMap:
    def test_python_detected(self) -> None:
        assert LANGUAGE_MAP[".py"] == "python"

    def test_typescript_detected(self) -> None:
        assert LANGUAGE_MAP[".ts"] == "typescript"
        assert LANGUAGE_MAP[".tsx"] == "typescript"

    def test_yaml_detected(self) -> None:
        assert LANGUAGE_MAP[".yml"] == "yaml"
        assert LANGUAGE_MAP[".yaml"] == "yaml"


# ── Token Counting ────────────────────────────────────────────────────────────


class TestTokenCounting:
    def test_empty_returns_zero(self) -> None:
        assert count_tokens("") == 0

    def test_short_string(self) -> None:
        # tiktoken cl100k_base: "hello world" = 2 tokens
        assert count_tokens("hello world") == 2

    def test_longer_code_more_tokens(self) -> None:
        short = "x = 1"
        longer = "def compute_ratio(a: float, b: float) -> float:\n    return a / b\n"
        assert count_tokens(longer) > count_tokens(short)


# ── Symbol Extraction ─────────────────────────────────────────────────────────


class TestSymbolExtraction:
    def test_extracts_function_names(self) -> None:
        content = "def foo():\n    pass\n\ndef bar(x):\n    return x\n"
        symbols = _extract_symbols(content)
        assert "foo" in symbols
        assert "bar" in symbols

    def test_extracts_class_names(self) -> None:
        content = "class MyService:\n    pass\n"
        symbols = _extract_symbols(content)
        assert "MyService" in symbols

    def test_extracts_async_functions(self) -> None:
        content = "async def fetch(url: str):\n    pass\n"
        symbols = _extract_symbols(content)
        assert "fetch" in symbols

    def test_empty_content_returns_empty(self) -> None:
        assert _extract_symbols("") == []

    def test_only_comments_returns_empty(self) -> None:
        content = "# This is a comment\n# Another one\n"
        assert _extract_symbols(content) == []


# ── BM25 Tokenizer ────────────────────────────────────────────────────────────


class TestBM25Tokenizer:
    def test_lowercase(self) -> None:
        tokens = _tokenize_for_bm25("UserService")
        assert "userservice" in tokens

    def test_splits_on_underscores(self) -> None:
        tokens = _tokenize_for_bm25("get_user_by_id")
        assert "get" in tokens
        assert "user" in tokens
        assert "by" in tokens
        assert "id" in tokens

    def test_splits_camel_case_path(self) -> None:
        # Path splitting on slashes
        tokens = _tokenize_for_bm25("src/services/user_service.py")
        assert "src" in tokens
        assert "services" in tokens
        assert "user" in tokens

    def test_filters_very_short_tokens(self) -> None:
        tokens = _tokenize_for_bm25("a.b.c")
        # Single characters should be filtered (len > 1 requirement)
        assert "a" not in tokens
        assert "b" not in tokens


# ── Load File Contexts (with real temp files) ─────────────────────────────────


class TestLoadFileContexts:
    @pytest.fixture
    def temp_repo(self, tmp_path: Path) -> Path:
        """Create a temp directory with sample Python files."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "utils.py").write_text("def helper(x: int) -> int:\n    return x * 2\n")
        (tmp_path / "src" / "service.py").write_text("class MyService:\n    def run(self):\n        pass\n")
        return tmp_path

    @pytest.mark.asyncio
    async def test_loads_existing_files(self, temp_repo: Path) -> None:
        from src.tools.filesystem import load_file_contexts

        with patch(
            "src.tools._repo_cache.get_local_repo_path",
            new_callable=AsyncMock,
            return_value=str(temp_repo),
        ):
            results = await load_file_contexts(
                repo="owner/repo",
                paths=["src/utils.py", "src/service.py"],
            )

        assert len(results) == 2
        paths = {r["path"] for r in results}
        assert "src/utils.py" in paths
        assert "src/service.py" in paths

    @pytest.mark.asyncio
    async def test_skips_nonexistent_files(self, temp_repo: Path) -> None:
        from src.tools.filesystem import load_file_contexts

        with patch(
            "src.tools._repo_cache.get_local_repo_path",
            new_callable=AsyncMock,
            return_value=str(temp_repo),
        ):
            results = await load_file_contexts(
                repo="owner/repo",
                paths=["nonexistent.py"],
            )

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_truncates_large_file(self, tmp_path: Path) -> None:
        from src.tools.filesystem import load_file_contexts

        # Create a file with many tokens
        large_content = ("def foo():\n    pass\n\n") * 500
        (tmp_path / "big.py").write_text(large_content)

        with patch(
            "src.tools._repo_cache.get_local_repo_path",
            new_callable=AsyncMock,
            return_value=str(tmp_path),
        ):
            results = await load_file_contexts(
                repo="owner/repo",
                paths=["big.py"],
                max_tokens_per_file=100,
            )

        assert len(results) == 1
        result = results[0]
        assert result["token_count"] <= 100
        assert "[TRUNCATED" in result["content"]

    @pytest.mark.asyncio
    async def test_detects_language_from_extension(self, tmp_path: Path) -> None:
        from src.tools.filesystem import load_file_contexts

        (tmp_path / "config.yml").write_text("key: value\n")

        with patch(
            "src.tools._repo_cache.get_local_repo_path",
            new_callable=AsyncMock,
            return_value=str(tmp_path),
        ):
            results = await load_file_contexts("owner/repo", ["config.yml"])

        assert results[0]["language"] == "yaml"


# ── Repo Tree ─────────────────────────────────────────────────────────────────


class TestListRepoTree:
    @pytest.mark.asyncio
    async def test_generates_tree_string(self, tmp_path: Path) -> None:
        from src.tools.filesystem import list_repo_tree

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_main.py").write_text("")

        with patch(
            "src.tools._repo_cache.get_local_repo_path",
            new_callable=AsyncMock,
            return_value=str(tmp_path),
        ):
            tree = await list_repo_tree("owner/repo")

        assert "src" in tree
        assert "tests" in tree
        assert "main.py" in tree

    @pytest.mark.asyncio
    async def test_ignores_pycache(self, tmp_path: Path) -> None:
        from src.tools.filesystem import list_repo_tree

        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "foo.pyc").write_text("")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("")

        with patch(
            "src.tools._repo_cache.get_local_repo_path",
            new_callable=AsyncMock,
            return_value=str(tmp_path),
        ):
            tree = await list_repo_tree("owner/repo")

        assert "__pycache__" not in tree
        assert "app.py" in tree

    @pytest.mark.asyncio
    async def test_list_repo_tree_depth_and_permission_error(self, tmp_path: Path) -> None:
        from src.tools.filesystem import list_repo_tree
        from unittest.mock import MagicMock

        (tmp_path / "deep").mkdir()
        (tmp_path / "deep" / "level1").mkdir()
        (tmp_path / "deep" / "level1" / "file.py").write_text("print(1)")

        with patch("src.tools._repo_cache.get_local_repo_path", new_callable=AsyncMock, return_value=str(tmp_path)):
            tree = await list_repo_tree("owner/repo", max_depth=0)
            assert "level1" not in tree

        # Test PermissionError handling
        mock_path = MagicMock()
        mock_path.iterdir.side_effect = PermissionError("no access")
        with patch("src.tools._repo_cache.get_local_repo_path", new_callable=AsyncMock, return_value=str(tmp_path)):
            with patch("pathlib.Path.iterdir", side_effect=PermissionError("no access")):
                tree2 = await list_repo_tree("owner/repo")
                assert tree2 == ""


class TestListTestFiles:
    @pytest.mark.asyncio
    async def test_list_test_files_finds_tests(self, tmp_path: Path) -> None:
        from src.tools.filesystem import list_test_files

        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_api.py").write_text("def test_x(): pass")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("def run(): pass")

        with patch("src.tools._repo_cache.get_local_repo_path", new_callable=AsyncMock, return_value=str(tmp_path)):
            test_files = await list_test_files("owner/repo")

        assert "tests/test_api.py" in test_files
        assert "src/app.py" not in test_files


class TestSearchFilesBm25:
    @pytest.mark.asyncio
    async def test_search_files_bm25(self, tmp_path: Path) -> None:
        from src.tools.filesystem import find_relevant_files

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "user_service.py").write_text("class UserService:\n    def get_user_by_id(self): pass")
        (tmp_path / "src" / "order_service.py").write_text("class OrderService:\n    def process_order(self): pass")
        (tmp_path / "src" / "payment_service.py").write_text("class PaymentService:\n    def charge(self): pass")
        (tmp_path / "src" / "catalog_service.py").write_text("class CatalogService:\n    def list_items(self): pass")
        (tmp_path / ".venv").mkdir()
        (tmp_path / ".venv" / "ignored.py").write_text("class Ignored: pass")

        with patch("src.tools._repo_cache.get_local_repo_path", new_callable=AsyncMock, return_value=str(tmp_path)):
            # Positive score with larger corpus
            results = await find_relevant_files("owner/repo", query="get_user_by_id UserService", exclude=[])
            assert "src/user_service.py" in results
            assert ".venv/ignored.py" not in results

            # Exclude works
            results_excluded = await find_relevant_files(
                "owner/repo", query="get_user_by_id", exclude=["src/user_service.py"]
            )
            assert "src/user_service.py" not in results_excluded

            # Read error handling during search
            with patch("pathlib.Path.read_text", side_effect=RuntimeError("read error")):
                res_err = await find_relevant_files("owner/repo", query="search", exclude=[])
                assert res_err == []

            # Empty corpus returns empty
            empty_results = await find_relevant_files(
                "owner/repo",
                query="something",
                exclude=[
                    "src/user_service.py",
                    "src/order_service.py",
                    "src/payment_service.py",
                    "src/catalog_service.py",
                ],
            )
            assert empty_results == []

            # 2-file small corpus (score == 0 fallback)
            small_results = await find_relevant_files(
                "owner/repo", query="get_user_by_id", exclude=["src/payment_service.py", "src/catalog_service.py"]
            )
            assert "src/user_service.py" in small_results

    @pytest.mark.asyncio
    async def test_load_file_contexts_read_exception(self, tmp_path: Path) -> None:
        from src.tools.filesystem import load_file_contexts

        f = tmp_path / "unreadable.py"
        f.write_text("hello")

        with patch("src.tools._repo_cache.get_local_repo_path", new_callable=AsyncMock, return_value=str(tmp_path)):
            with patch("pathlib.Path.read_text", side_effect=RuntimeError("disk err")):
                res = await load_file_contexts("owner/repo", ["unreadable.py"])
                assert res == []
