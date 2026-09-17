"""Unit tests for src/tools/search.py with 100% statement coverage."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.tools.search import (
    CodeSymbol,
    extract_ast_symbols,
    find_relevant_files,
    search_symbols,
    tokenize_for_search,
)


def test_code_symbol_to_dict():
    sym = CodeSymbol(name="User", kind="class", line_number=12, docstring="User model")
    d = sym.to_dict()
    assert d == {
        "name": "User",
        "kind": "class",
        "line_number": 12,
        "docstring": "User model",
    }


def test_extract_ast_symbols_python():
    code = '''"""Module doc."""
class UserService:
    """Service class."""
    pass

async def fetch_user(user_id: str):
    """Fetch user doc."""
    pass

def compute_total():
    pass
'''
    symbols = extract_ast_symbols(code)
    names = [s.name for s in symbols]
    assert "UserService" in names
    assert "fetch_user" in names
    assert "compute_total" in names

    kinds = {s.name: s.kind for s in symbols}
    assert kinds["UserService"] == "class"
    assert kinds["fetch_user"] == "async_function"
    assert kinds["compute_total"] == "function"


def test_extract_ast_symbols_syntax_error_regex_fallback():
    broken_code = """
class InvalidClass:::
    def broken_func(
"""
    symbols = extract_ast_symbols(broken_code)
    names = [s.name for s in symbols]
    assert "InvalidClass" in names or "broken_func" in names


def test_tokenize_for_search():
    tokens = tokenize_for_search("calculate_total_price and getInvoiceDetails with 123-abc")
    assert "calculate" in tokens
    assert "price" in tokens
    assert "calculatetotalprice" in tokens
    assert "invoice" in tokens
    assert "details" in tokens


@pytest.mark.asyncio
async def test_find_relevant_files_search(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "service.py").write_text(
        "class BillingEngine:\n"
        '    """Handles payments and invoicing."""\n'
        "    async def process_bill(self): pass\n"
    )
    (tmp_path / "src" / "auth.py").write_text(
        'class AuthManager:\n    """Handles tokens."""\n    def login(self): pass\n'
    )
    (tmp_path / "src" / "empty.py").write_text("# Just a comment\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "ignored.py").write_text("class Ignored: pass")

    with patch(
        "src.tools._repo_cache.get_local_repo_path",
        new_callable=AsyncMock,
        return_value=str(tmp_path),
    ):
        # Exact symbol boost match
        matches = await find_relevant_files(
            "owner/repo", query="BillingEngine invoice payments", exclude=[]
        )
        assert "src/service.py" in matches
        assert ".venv/ignored.py" not in matches

        # Exclude works
        matches_ex = await find_relevant_files(
            "owner/repo", query="BillingEngine", exclude=["src/service.py"]
        )
        assert "src/service.py" not in matches_ex

        # Empty corpus
        empty = await find_relevant_files(
            "owner/repo", query="test", exclude=["src/service.py", "src/auth.py", "src/empty.py"]
        )
        assert empty == []

        # Zero score query fallback
        small = await find_relevant_files(
            "owner/repo", query="service", exclude=["src/auth.py", "src/empty.py"]
        )
        assert "src/service.py" in small

        # File read exception swallowed
        with patch("pathlib.Path.read_text", side_effect=RuntimeError("read failed")):
            err_res = await find_relevant_files("owner/repo", query="BillingEngine", exclude=[])
            assert err_res == []


@pytest.mark.asyncio
async def test_search_symbols(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "math_utils.py").write_text(
        "class Matrix:\n    pass\n\n"
        "def calculate_dot_product():\n"
        "    '''Compute dot product'''\n    pass\n"
    )
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hook.py").write_text("def hook(): pass")

    with patch(
        "src.tools._repo_cache.get_local_repo_path",
        new_callable=AsyncMock,
        return_value=str(tmp_path),
    ):
        results = await search_symbols("owner/repo", symbol_query="dot_product")
        assert len(results) == 1
        assert results[0]["name"] == "calculate_dot_product"
        assert results[0]["kind"] == "function"
        assert results[0]["file"] == "pkg/math_utils.py"

        # Search by docstring
        doc_results = await search_symbols("owner/repo", symbol_query="dot product")
        assert len(doc_results) >= 1

        # File read error handled
        with patch("pathlib.Path.read_text", side_effect=OSError("disk error")):
            res_err = await search_symbols("owner/repo", symbol_query="anything")
            assert res_err == []
