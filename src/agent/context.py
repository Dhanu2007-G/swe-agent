"""
src/agent/context.py — AST-aware context builder for the coder node.

This is the component that separates toy agents from production ones.
Instead of dumping raw file contents into the LLM context, we:
  1. Parse each file with tree-sitter to extract structural information
  2. Apply a token budget — never exceed the model's context window
  3. Rank files by relevance to the current task
  4. Truncate intelligently at function/class boundaries, not mid-line

The roadmap calls this out explicitly: "Extract function signatures and class
definitions. Give the LLM structure-aware context, not raw file dumps."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import structlog
import tiktoken

from src.config import get_settings

log = structlog.get_logger(__name__)


def _get_tokenizer() -> Any:
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:

        class _FallbackTokenizer:
            def encode(self, text: str, disallowed_special: tuple[str, ...] = ()) -> list[int]:
                if not text:
                    return []
                import re

                tokens = re.findall(r"\w+|[^\w\s]", text)
                return list(range(len(tokens)))

        return _FallbackTokenizer()


_TOKENIZER = _get_tokenizer()


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text, disallowed_special=()))


# ── Data Classes ──────────────────────────────────────────────────────────────


@dataclass
class Symbol:
    """A named code symbol extracted by AST parsing."""

    name: str
    kind: str  # "function" | "class" | "method" | "import"
    start_line: int
    end_line: int
    docstring: str = ""
    signature: str = ""  # e.g. "def get_user(user_id: int) -> User | None:"


@dataclass
class FileContext:
    """A file prepared for LLM consumption, with token budget applied."""

    path: str
    language: str
    content: str  # possibly truncated
    symbols: list[Symbol] = field(default_factory=list)
    token_count: int = 0
    was_truncated: bool = False
    truncation_note: str = ""

    def to_prompt_block(self) -> str:
        """Format for insertion into a coder/corrector prompt."""
        lines = [f"### {self.path}"]
        if self.symbols:
            sym_summary = ", ".join(f"{s.kind} `{s.name}`" for s in self.symbols[:8])
            lines.append(f"*Symbols: {sym_summary}*")
        if self.was_truncated:
            lines.append(f"*{self.truncation_note}*")
        lines.append(f"```{self.language}")
        lines.append(self.content)
        lines.append("```")
        return "\n".join(lines)


@dataclass
class ContextBudget:
    """Token budget tracker across all files."""

    total_limit: int
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.total_limit - self.used)

    def consume(self, tokens: int) -> bool:
        """Returns True if budget allows; False if would exceed."""
        if tokens > self.remaining:
            return False
        self.used += tokens
        return True

    @property
    def utilization_pct(self) -> float:
        return (self.used / self.total_limit) * 100


# ── Main Context Builder ──────────────────────────────────────────────────────


class ContextBuilder:
    """
    Builds a token-budgeted, AST-enriched context package for the coder node.

    Usage:
        builder = ContextBuilder(task_description="Add None guard in get_user")
        for path in planned_files:
            await builder.add_file(path, content, language)
        context_blocks = builder.render()
    """

    def __init__(
        self,
        task_description: str,
        token_limit: int | None = None,
    ) -> None:
        self.task_description = task_description
        settings = get_settings()
        self._budget = ContextBudget(total_limit=token_limit or settings.agent_max_context_tokens)
        self._files: list[FileContext] = []

    def add_file(
        self,
        path: str,
        content: str,
        language: str = "python",
        priority: int = 0,  # higher = added first, gets more budget
    ) -> bool:
        """
        Add a file to the context. Returns False if budget is exhausted.
        Files are truncated at symbol boundaries if needed.
        """
        symbols = _extract_symbols_ast(content, language)
        processed = _apply_token_budget(
            path=path,
            content=content,
            language=language,
            symbols=symbols,
            budget=self._budget,
        )

        if processed is None:
            log.warning(
                "context.file_skipped_no_budget",
                path=path,
                remaining_tokens=self._budget.remaining,
            )
            return False

        self._files.append(processed)
        log.debug(
            "context.file_added",
            path=path,
            tokens=processed.token_count,
            truncated=processed.was_truncated,
            budget_remaining=self._budget.remaining,
        )
        return True

    def render(self) -> str:
        """Render all files into a single prompt-ready string."""
        if not self._files:
            return "No files loaded into context."

        blocks = [f.to_prompt_block() for f in self._files]
        summary = (
            f"*Context: {len(self._files)} file(s), "
            f"{self._budget.used:,} / {self._budget.total_limit:,} tokens "
            f"({self._budget.utilization_pct:.1f}% budget used)*"
        )
        return summary + "\n\n" + "\n\n".join(blocks)

    def get_symbol_index(self) -> dict[str, list[str]]:
        """Return {file_path: [symbol_names]} for debugging/logging."""
        return {fc.path: [s.name for s in fc.symbols] for fc in self._files}

    @property
    def file_count(self) -> int:
        return len(self._files)

    @property
    def total_tokens(self) -> int:
        return self._budget.used


# ── Token Budget Application ──────────────────────────────────────────────────


def _apply_token_budget(
    path: str,
    content: str,
    language: str,
    symbols: list[Symbol],
    budget: ContextBudget,
) -> FileContext | None:
    """
    Fit the file into the remaining token budget.
    Truncation strategy (in order of preference):
      1. Full file — ideal
      2. Truncate at a function/class boundary
      3. Hard truncate at budget limit
      4. Return None if even the header + signature won't fit
    """
    full_tokens = _count_tokens(content)

    # Case 1: fits in full
    if budget.consume(full_tokens):
        return FileContext(
            path=path,
            language=language,
            content=content,
            symbols=symbols,
            token_count=full_tokens,
        )

    # Case 2: truncate at symbol boundary
    remaining = budget.remaining
    if remaining < 100:
        return None  # not enough budget for even a stub

    # Find the best symbol boundary to cut at
    truncated_content, cut_note = _truncate_at_boundary(
        content=content,
        symbols=symbols,
        max_tokens=remaining,
    )
    truncated_tokens = _count_tokens(truncated_content)
    if not budget.consume(truncated_tokens):
        return None

    return FileContext(
        path=path,
        language=language,
        content=truncated_content,
        symbols=symbols,
        token_count=truncated_tokens,
        was_truncated=True,
        truncation_note=cut_note,
    )


def _truncate_at_boundary(
    content: str,
    symbols: list[Symbol],
    max_tokens: int,
) -> tuple[str, str]:
    """
    Try to cut the file at the end of a complete function or class.
    Falls back to a hard character truncation if no clean boundary is found.
    """
    lines = content.splitlines(keepends=True)

    # Walk backwards through symbols to find the last one that fits
    for symbol in reversed(symbols):
        # Include up to the end of this symbol
        partial = "".join(lines[: symbol.end_line + 1])
        if _count_tokens(partial) <= max_tokens:
            remaining_symbols = len(symbols) - symbols.index(symbol) - 1
            note = f"Truncated after `{symbol.name}` ({remaining_symbols} more symbol(s) omitted)"
            return partial, note

    # No symbol boundary fits — try line-by-line truncation
    accumulated_lines: list[str] = []
    for line in lines:
        test_chunk = "".join(accumulated_lines + [line])
        if _count_tokens(test_chunk) <= max_tokens:
            accumulated_lines.append(line)
        else:
            break

    if accumulated_lines:
        return "".join(accumulated_lines), "Hard-truncated to fit token budget"

    # If even a single line doesn't fit, character truncate
    chars_per_token = max(1, len(content) // max(_count_tokens(content), 1))
    char_limit = max_tokens * chars_per_token
    return content[:char_limit], "Hard-truncated to fit token budget"


# ── Tree-sitter AST Extraction ────────────────────────────────────────────────

# Module-level cached tree-sitter parsers (lazy-initialized per language)
_TS_PARSER = None
_TS_PARSERS: dict[str, Any] = {}

_GRAMMAR_MODULES: dict[str, tuple[str, str]] = {
    "python": ("tree_sitter_python", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "javascript": ("tree_sitter_javascript", "language"),
    "go": ("tree_sitter_go", "language"),
    "rust": ("tree_sitter_rust", "language"),
    "java": ("tree_sitter_java", "language"),
}


def _get_treesitter_parser(language: str = "python") -> Any:
    """Return a cached tree-sitter parser for the given language."""
    global _TS_PARSER, _TS_PARSERS
    if language == "python" and _TS_PARSER is not None:
        return _TS_PARSER
    if language in _TS_PARSERS:
        return _TS_PARSERS[language]

    if language not in _GRAMMAR_MODULES:
        raise ImportError(f"No tree-sitter grammar configured for {language}")

    module_name, func_name = _GRAMMAR_MODULES[language]
    try:
        import importlib

        from tree_sitter import Language, Parser

        mod = importlib.import_module(module_name)
        lang_fn = getattr(mod, func_name)
        lang_obj = Language(lang_fn())
        parser = Parser(lang_obj)
        _TS_PARSERS[language] = parser
        if language == "python":
            _TS_PARSER = parser
        log.debug("context.treesitter_parser_cached", language=language)
        return parser
    except ImportError:
        log.warning("context.treesitter_not_installed", language=language)
        raise


def _extract_symbols_ast(content: str, language: str) -> list[Symbol]:
    """
    Extract symbols using tree-sitter when available, regex as fallback.
    tree-sitter gives us accurate line ranges for smart truncation.
    """
    if language == "python":
        try:
            return _extract_python_symbols_treesitter(content)
        except Exception as e:
            log.debug("context.treesitter_failed", language=language, error=str(e))
            return _extract_python_symbols_regex(content)

    if language in _GRAMMAR_MODULES:
        try:
            parser = _get_treesitter_parser(language)
            return _extract_generic_symbols_treesitter(content, parser, language)
        except Exception as e:
            log.debug("context.treesitter_failed", language=language, error=str(e))

    return _extract_polyglot_symbols_regex(content, language)


def _extract_generic_symbols_treesitter(content: str, parser: Any, language: str) -> list[Symbol]:
    """Extract AST symbols for polyglot languages using tree-sitter."""
    tree = parser.parse(content.encode())
    symbols: list[Symbol] = []
    target_types = {
        "function_declaration",
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "type_declaration",
        "function_item",
        "struct_item",
        "enum_item",
        "trait_item",
    }

    def _walk(node: Any) -> None:
        if node.type in target_types:
            name_node = node.child_by_field_name("name")
            name = name_node.text.decode() if name_node else "unknown"
            kind = "class" if ("class" in node.type or "struct" in node.type) else "function"
            symbols.append(
                Symbol(
                    name=name,
                    kind=kind,
                    start_line=node.start_point[0],
                    end_line=node.end_point[0],
                    signature=f"{kind} {name}",
                )
            )
        for child in node.children:
            _walk(child)

    _walk(tree.root_node)
    return symbols


def _extract_python_symbols_treesitter(content: str) -> list[Symbol]:
    """
    Use tree-sitter-python for accurate AST-based extraction.
    Returns functions, classes, and methods with line ranges.
    Parser is cached at module level for performance.
    """
    parser = _get_treesitter_parser()
    tree = parser.parse(content.encode())
    symbols: list[Symbol] = []

    def _walk(node: Any, class_name: str = "") -> None:
        if node.type in ("function_definition", "async_function_definition"):
            name_node = node.child_by_field_name("name")
            params_node = node.child_by_field_name("parameters")
            return_node = node.child_by_field_name("return_type")

            name = name_node.text.decode() if name_node else "unknown"
            params = params_node.text.decode() if params_node else "()"
            ret = (": " + return_node.text.decode()) if return_node else ""

            kind = "method" if class_name else "function"
            prefix = "async def" if node.type == "async_function_definition" else "def"
            full_name = f"{class_name}.{name}" if class_name else name

            # Extract docstring from first expression statement
            docstring = ""
            body = node.child_by_field_name("body")
            if body and body.child_count > 0:
                first = body.children[0]
                if first.type == "expression_statement":
                    child = first.children[0] if first.child_count > 0 else None
                    if child and child.type == "string":
                        docstring = child.text.decode().strip("\"'").strip()[:200]

            symbols.append(
                Symbol(
                    name=full_name,
                    kind=kind,
                    start_line=node.start_point[0],
                    end_line=node.end_point[0],
                    signature=f"{prefix} {name}{params}{ret}:",
                    docstring=docstring,
                )
            )

        elif node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            name = name_node.text.decode() if name_node else "unknown"
            symbols.append(
                Symbol(
                    name=name,
                    kind="class",
                    start_line=node.start_point[0],
                    end_line=node.end_point[0],
                    signature=f"class {name}:",
                )
            )
            for child in node.children:
                _walk(child, class_name=name)
            return  # already recursed

        for child in node.children:
            _walk(child, class_name=class_name)

    _walk(tree.root_node)
    return symbols


def _extract_python_symbols_regex(content: str) -> list[Symbol]:
    """Regex fallback — less accurate line ranges, but no tree-sitter dependency."""
    symbols: list[Symbol] = []
    lines = content.splitlines()

    for i, line in enumerate(lines):
        # Functions and methods
        m = re.match(r"^(\s*)(async\s+)?def\s+([a-zA-Z_]\w*)\s*\(([^)]*)\)", line)
        if m:
            indent = len(m.group(1))
            name = m.group(3)
            params = m.group(4)
            kind = "method" if indent > 0 else "function"
            prefix = "async def" if m.group(2) else "def"
            symbols.append(
                Symbol(
                    name=name,
                    kind=kind,
                    start_line=i,
                    end_line=_find_block_end(lines, i),
                    signature=f"{prefix} {name}({params}):",
                )
            )
            continue

        # Classes
        m = re.match(r"^class\s+([a-zA-Z_]\w*)", line)
        if m:
            name = m.group(1)
            symbols.append(
                Symbol(
                    name=name,
                    kind="class",
                    start_line=i,
                    end_line=_find_block_end(lines, i),
                    signature=f"class {name}:",
                )
            )

    return symbols


def _find_brace_block_end(lines: list[str], start: int) -> int:
    """Find the matching closing brace line for a block starting at `start`."""
    open_count = 0
    found_any = False
    for i in range(start, len(lines)):
        line = lines[i]
        open_count += line.count("{") - line.count("}")
        if "{" in line:
            found_any = True
        if found_any and open_count <= 0:
            return i
    return min(start + 20, len(lines) - 1)


def _extract_polyglot_symbols_regex(content: str, language: str) -> list[Symbol]:
    """Rich symbol extraction for polyglot languages using regex."""
    symbols: list[Symbol] = []
    lang = language.lower()
    lines = content.splitlines()

    patterns: list[tuple[str, str]] = []
    if lang in ("javascript", "typescript", "jsx", "tsx"):
        patterns = [
            (r"(?:export\s+)?(?:default\s+)?class\s+(\w+)", "class"),
            (r"(?:export\s+)?interface\s+(\w+)", "interface"),
            (r"(?:export\s+)?type\s+(\w+)\s*=", "type"),
            (r"(?:export\s+)?(?:async\s+)?function\s+(\w+)", "function"),
            (r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(", "function"),
        ]
    elif lang == "go":
        patterns = [
            (r"^type\s+(\w+)\s+struct", "struct"),
            (r"^type\s+(\w+)\s+interface", "interface"),
            (r"^func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(", "function"),
        ]
    elif lang == "rust":
        patterns = [
            (r"^\s*(?:pub\s+)?struct\s+(\w+)", "struct"),
            (r"^\s*(?:pub\s+)?enum\s+(\w+)", "enum"),
            (r"^\s*(?:pub\s+)?trait\s+(\w+)", "trait"),
            (r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", "function"),
        ]
    elif lang == "java":
        patterns = [
            (r"(?:public|protected|private)?\s*(?:static\s+)?class\s+(\w+)", "class"),
            (r"(?:public|protected|private)?\s*(?:static\s+)?interface\s+(\w+)", "interface"),
            (r"(?:public|protected|private|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\(", "method"),
        ]

    for i, line in enumerate(lines):
        for pat, kind in patterns:
            m = re.search(pat, line)
            if m:
                name = m.group(1)
                end_line = _find_brace_block_end(lines, i)
                symbols.append(
                    Symbol(
                        name=name,
                        kind=kind,
                        start_line=i,
                        end_line=end_line,
                        signature=line.strip()[:80],
                    )
                )
                break

    return symbols


_extract_generic_symbols_regex = _extract_polyglot_symbols_regex


def _find_block_end(lines: list[str], start: int) -> int:
    """Find the last line of an indented block starting at `start`."""
    if start >= len(lines):
        return start
    base_indent = len(lines[start]) - len(lines[start].lstrip())
    for i in range(start + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            continue
        line_indent = len(lines[i]) - len(lines[i].lstrip())
        if line_indent <= base_indent and stripped:
            return i - 1
    return len(lines) - 1


# ── Convenience builder from raw dicts ───────────────────────────────────────


def build_context_from_file_dicts(
    file_dicts: list[dict[str, Any]],
    task_description: str,
    token_limit: int | None = None,
) -> str:
    """
    Entry point called from code_node and correct_node.
    Takes the raw file context dicts from load_file_contexts() and returns
    a fully formatted, token-budgeted context string.
    """
    builder = ContextBuilder(
        task_description=task_description,
        token_limit=token_limit,
    )

    for fc in file_dicts:
        builder.add_file(
            path=fc["path"],
            content=fc["content"],
            language=fc.get("language", "python"),
        )

    log.info(
        "context.built",
        files=builder.file_count,
        tokens=builder.total_tokens,
        symbols=builder.get_symbol_index(),
    )
    return builder.render()
