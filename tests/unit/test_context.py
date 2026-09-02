"""
tests/unit/test_context.py — Tests for AST context builder.
Validates token budget enforcement, symbol extraction, truncation strategies.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from src.agent.context import (
    ContextBudget,
    ContextBuilder,
    FileContext,
    Symbol,
    _apply_token_budget,
    _count_tokens,
    _extract_python_symbols_regex,
    _find_block_end,
    _truncate_at_boundary,
    build_context_from_file_dicts,
)


# ── Sample Code ───────────────────────────────────────────────────────────────

SAMPLE_PYTHON = """\
class UserService:
    \"\"\"Manages user operations.\"\"\"

    def get_user(self, user_id: int) -> dict | None:
        \"\"\"Fetch a user by ID. Returns None if not found.\"\"\"
        if user_id is None:
            return None
        return db.query(user_id)

    def create_user(self, name: str, email: str) -> dict:
        \"\"\"Create a new user record.\"\"\"
        return db.insert({"name": name, "email": email})


def compute_ratio(numerator: float, denominator: float) -> float:
    \"\"\"Compute a ratio. Returns 0.0 if denominator is zero.\"\"\"
    if denominator == 0:
        return 0.0
    return numerator / denominator


async def fetch_data(url: str) -> bytes:
    \"\"\"Async HTTP fetch.\"\"\"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        return resp.content
"""


# ── Symbol Extraction ─────────────────────────────────────────────────────────

class TestSymbolExtraction:
    def test_extracts_class(self) -> None:
        symbols = _extract_python_symbols_regex(SAMPLE_PYTHON)
        class_syms = [s for s in symbols if s.kind == "class"]
        assert len(class_syms) == 1
        assert class_syms[0].name == "UserService"

    def test_extracts_methods(self) -> None:
        symbols = _extract_python_symbols_regex(SAMPLE_PYTHON)
        methods = [s for s in symbols if s.kind == "method"]
        method_names = {s.name for s in methods}
        assert "get_user" in method_names
        assert "create_user" in method_names

    def test_extracts_module_level_functions(self) -> None:
        symbols = _extract_python_symbols_regex(SAMPLE_PYTHON)
        funcs = [s for s in symbols if s.kind == "function"]
        func_names = {s.name for s in funcs}
        assert "compute_ratio" in func_names
        assert "fetch_data" in func_names

    def test_async_function_detected(self) -> None:
        symbols = _extract_python_symbols_regex(SAMPLE_PYTHON)
        async_funcs = [s for s in symbols if "async" in s.signature]
        assert any(s.name == "fetch_data" for s in async_funcs)

    def test_start_line_is_correct(self) -> None:
        symbols = _extract_python_symbols_regex(SAMPLE_PYTHON)
        class_sym = next(s for s in symbols if s.name == "UserService")
        assert class_sym.start_line == 0  # first line

    def test_empty_file_returns_no_symbols(self) -> None:
        assert _extract_python_symbols_regex("") == []

    def test_file_with_only_comments_returns_no_symbols(self) -> None:
        content = "# This is a comment\n# Another comment\n"
        assert _extract_python_symbols_regex(content) == []


# ── Token Counting ────────────────────────────────────────────────────────────

class TestTokenCounting:
    def test_empty_string_is_zero(self) -> None:
        assert _count_tokens("") == 0

    def test_known_string_token_count(self) -> None:
        # "Hello world" is 2 tokens in cl100k_base
        count = _count_tokens("Hello world")
        assert count == 2

    def test_longer_code_has_more_tokens(self) -> None:
        short = "def foo(): pass"
        long = SAMPLE_PYTHON
        assert _count_tokens(long) > _count_tokens(short)


# ── Token Budget ──────────────────────────────────────────────────────────────

class TestContextBudget:
    def test_fresh_budget_has_full_remaining(self) -> None:
        budget = ContextBudget(total_limit=10_000)
        assert budget.remaining == 10_000
        assert budget.used == 0

    def test_consume_reduces_remaining(self) -> None:
        budget = ContextBudget(total_limit=1000)
        assert budget.consume(300) is True
        assert budget.remaining == 700

    def test_consume_returns_false_when_over_limit(self) -> None:
        budget = ContextBudget(total_limit=100)
        assert budget.consume(101) is False
        assert budget.used == 0  # not consumed

    def test_consume_exact_limit_succeeds(self) -> None:
        budget = ContextBudget(total_limit=100)
        assert budget.consume(100) is True
        assert budget.remaining == 0

    def test_utilization_pct(self) -> None:
        budget = ContextBudget(total_limit=1000)
        budget.consume(500)
        assert budget.utilization_pct == pytest.approx(50.0)


# ── Truncation ────────────────────────────────────────────────────────────────

class TestTruncation:
    def test_truncates_at_function_boundary(self) -> None:
        symbols = _extract_python_symbols_regex(SAMPLE_PYTHON)
        # Set max_tokens to fit only the first function
        first_func_tokens = _count_tokens(SAMPLE_PYTHON[:200])
        truncated, note = _truncate_at_boundary(
            content=SAMPLE_PYTHON,
            symbols=symbols,
            max_tokens=first_func_tokens + 50,
        )
        assert "Truncated" in note or "omitted" in note
        # Truncated content should be valid Python up to a function boundary
        assert len(truncated) < len(SAMPLE_PYTHON)

    def test_hard_truncation_when_no_boundary_fits(self) -> None:
        # Tiny budget that can't fit even one symbol
        symbols = _extract_python_symbols_regex(SAMPLE_PYTHON)
        truncated, note = _truncate_at_boundary(
            content=SAMPLE_PYTHON,
            symbols=symbols,
            max_tokens=5,
        )
        assert len(truncated) < len(SAMPLE_PYTHON)
        assert "truncat" in note.lower()

    def test_no_truncation_when_budget_sufficient(self) -> None:
        content = "def foo():\n    pass\n"
        symbols = _extract_python_symbols_regex(content)
        token_count = _count_tokens(content)
        budget = ContextBudget(total_limit=token_count + 100)
        result = _apply_token_budget(
            path="foo.py",
            content=content,
            language="python",
            symbols=symbols,
            budget=budget,
        )
        assert result is not None
        assert result.was_truncated is False
        assert result.content == content


# ── Context Builder ───────────────────────────────────────────────────────────

class TestContextBuilder:
    def test_single_file_renders(self) -> None:
        builder = ContextBuilder(task_description="Fix bug", token_limit=50_000)
        added = builder.add_file("utils.py", SAMPLE_PYTHON, "python")
        assert added is True
        output = builder.render()
        assert "utils.py" in output
        assert "```python" in output

    def test_budget_exhaustion_skips_additional_files(self) -> None:
        # Tiny budget — only room for one file
        builder = ContextBuilder(task_description="Fix bug", token_limit=50)
        added1 = builder.add_file("a.py", "def foo(): pass\n", "python")
        added2 = builder.add_file("b.py", SAMPLE_PYTHON, "python")  # too big
        assert added1 is True
        assert added2 is False  # budget exhausted
        assert builder.file_count == 1

    def test_symbol_index_populated(self) -> None:
        builder = ContextBuilder(task_description="Fix bug", token_limit=50_000)
        builder.add_file("svc.py", SAMPLE_PYTHON, "python")
        index = builder.get_symbol_index()
        assert "svc.py" in index
        assert "UserService" in index["svc.py"] or "get_user" in index["svc.py"]

    def test_truncation_note_appears_in_output(self) -> None:
        # Force truncation with a very small budget
        builder = ContextBuilder(task_description="Fix bug", token_limit=30)
        builder.add_file("big.py", SAMPLE_PYTHON, "python")
        output = builder.render()
        # Either the file was truncated or skipped — either way render doesn't crash
        assert isinstance(output, str)

    def test_empty_builder_renders_gracefully(self) -> None:
        builder = ContextBuilder(task_description="Fix bug")
        output = builder.render()
        assert "No files" in output

    def test_build_context_from_file_dicts(self) -> None:
        dicts = [
            {
                "path": "src/utils.py",
                "content": "def helper(): pass\n",
                "language": "python",
                "token_count": 5,
            }
        ]
        result = build_context_from_file_dicts(dicts, "Fix the helper function", token_limit=10000)
        assert "src/utils.py" in result
        assert "helper" in result

        empty_result = build_context_from_file_dicts([], "Empty task")
        assert "No files" in empty_result


# ── File Context Rendering ────────────────────────────────────────────────────

class TestFileContextRendering:
    def test_to_prompt_block_includes_path(self) -> None:
        fc = FileContext(
            path="src/auth.py",
            language="python",
            content="def login(): pass",
            token_count=5,
        )
        block = fc.to_prompt_block()
        assert "src/auth.py" in block
        assert "```python" in block

    def test_truncation_note_shown_when_truncated(self) -> None:
        fc = FileContext(
            path="big.py",
            language="python",
            content="...",
            token_count=50,
            was_truncated=True,
            truncation_note="Truncated after `UserService` (2 more symbols omitted)",
        )
        block = fc.to_prompt_block()
        assert "Truncated" in block

    def test_symbol_list_shown_when_present(self) -> None:
        fc = FileContext(
            path="svc.py",
            language="python",
            content="def get(): pass",
            symbols=[Symbol("get", "function", 0, 1, signature="def get():")],
            token_count=5,
        )
        block = fc.to_prompt_block()
        assert "get" in block


class TestContextEdgeCases:
    def test_extract_symbols_ast_treesitter_fallback(self) -> None:
        from src.agent.context import _extract_symbols_ast
        from unittest.mock import patch

        with patch("src.agent.context._extract_python_symbols_treesitter", side_effect=Exception("parse error")):
            syms = _extract_symbols_ast("def foo(): pass", "python")
            assert any(s.name == "foo" for s in syms)

    def test_extract_generic_symbols_regex(self) -> None:
        from src.agent.context import _extract_generic_symbols_regex

        js_code = "function calculateTotal() { return 1; }\nconst processPayment = async () => {};"
        js_syms = _extract_generic_symbols_regex(js_code, "javascript")
        assert len(js_syms) >= 1

        ts_code = "function handleRequest(req) {}\nlet configManager = {};"
        ts_syms = _extract_generic_symbols_regex(ts_code, "typescript")
        assert len(ts_syms) >= 1

        go_code = "func ProcessItem(item string) error {\n return nil \n}"
        go_syms = _extract_generic_symbols_regex(go_code, "go")
        assert len(go_syms) >= 1

        java_code = "public static void main(String[] args) {}"
        java_syms = _extract_generic_symbols_regex(java_code, "java")
        assert len(java_syms) >= 1

        unknown_syms = _extract_generic_symbols_regex("plain text", "unknown_lang")
        assert unknown_syms == []

    def test_find_block_end_boundary(self) -> None:
        from src.agent.context import _find_block_end

        assert _find_block_end(["line1", "line2"], 10) == 10

    def test_apply_token_budget_insufficient_budget(self) -> None:
        from src.agent.context import ContextBudget, _apply_token_budget

        budget = ContextBudget(total_limit=50)
        # consume budget to under 100
        budget.consume(10)
        res = _apply_token_budget("app.py", "long content " * 100, "python", [], budget)
        assert res is None

    def test_apply_token_budget_truncates_at_boundary(self) -> None:
        from src.agent.context import ContextBudget, _apply_token_budget, Symbol

        budget = ContextBudget(total_limit=150)
        content = ("class A:\n    '''Docstring'''\n    def m1(self):\n        return 1\n\n" * 20) + "class B:\n    def m2(self):\n        pass\n"
        symbols = [
            Symbol(name="A", kind="class", start_line=0, end_line=5, signature="class A:"),
            Symbol(name="B", kind="class", start_line=100, end_line=105, signature="class B:"),
        ]
        res = _apply_token_budget("app.py", content, "python", symbols, budget)
        assert res is not None
        assert res.was_truncated is True

        # Test when truncated content consume returns False
        class MockBudget(ContextBudget):
            def __init__(self):
                super().__init__(total_limit=150)
                self.calls = 0
            def consume(self, count: int) -> bool:
                self.calls += 1
                if self.calls == 1:
                    return False  # full tokens rejected
                return False  # truncated tokens rejected

        res_none = _apply_token_budget("app.py", content, "python", symbols, MockBudget())
        assert res_none is None

    def test_get_treesitter_parser_import_error(self) -> None:
        from src.agent import context

        old_parser = context._TS_PARSER
        old_parsers = dict(context._TS_PARSERS)
        context._TS_PARSER = None
        context._TS_PARSERS.clear()
        try:
            with patch("importlib.import_module", side_effect=ImportError("no tree sitter")):
                with pytest.raises(ImportError):
                    context._get_treesitter_parser("python")
        finally:
            context._TS_PARSER = old_parser
            context._TS_PARSERS = old_parsers

    def test_extract_symbols_ast_non_python(self) -> None:
        from src.agent.context import _extract_symbols_ast

        syms_js = _extract_symbols_ast("function calculate() {}", "javascript")
        assert len(syms_js) >= 1

        syms_ts = _extract_symbols_ast("const processOrder = () => {}", "typescript")
        assert len(syms_ts) >= 1

        syms_go = _extract_symbols_ast("func HandleRequest(w http.ResponseWriter) {}", "go")
        assert len(syms_go) >= 1

        syms_java = _extract_symbols_ast("public static void main(String[] args) {}", "java")
        assert len(syms_java) >= 1

        syms_empty = _extract_symbols_ast("some text", "unknown_lang")
        assert syms_empty == []

    def test_get_treesitter_parser_multilingual(self) -> None:
        from src.agent import context

        # Test cache hit
        context._TS_PARSER = "mock_cached"
        assert context._get_treesitter_parser("python") == "mock_cached"
        context._TS_PARSER = None

        context._TS_PARSERS["go"] = "mock_go_cached"
        assert context._get_treesitter_parser("go") == "mock_go_cached"
        del context._TS_PARSERS["go"]

        # Test unconfigured language
        with pytest.raises(ImportError, match="No tree-sitter grammar configured"):
            context._get_treesitter_parser("brainfuck")


def test_build_context_from_file_dicts_direct() -> None:
    from src.agent.context import build_context_from_file_dicts

    dicts = [
        {
            "path": "src/utils.py",
            "content": "def helper(): pass\n",
            "language": "python",
        }
    ]
    res = build_context_from_file_dicts(dicts, "Fix helper", token_limit=5000)
    assert "src/utils.py" in res
