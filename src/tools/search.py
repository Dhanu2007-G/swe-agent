"""
src/tools/search.py — AST-aware Symbol + BM25 Hybrid Code Search Engine.
Combines structural AST symbol extraction (classes, functions, async defs, docstrings)
with BM25 lexical ranking to accurately locate relevant files and code symbols.
"""

from __future__ import annotations

import ast
import asyncio
import re
from pathlib import Path
from typing import Any

import structlog
from rank_bm25 import BM25Okapi

log = structlog.get_logger(__name__)

IGNORE_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".env",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
    "*.egg-info",
}


POLYGLOT_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
}

MANIFEST_FILES = {
    "package.json",
    "go.mod",
    "Cargo.toml",
    "pom.xml",
    "build.gradle",
}


class CodeSymbol:
    """Represents a code symbol extracted via AST."""

    def __init__(
        self,
        name: str,
        kind: str,  # 'class', 'function', 'async_function', 'method'
        line_number: int,
        docstring: str = "",
    ):
        self.name = name
        self.kind = kind
        self.line_number = line_number
        self.docstring = docstring

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "line_number": self.line_number,
            "docstring": self.docstring,
        }


def extract_ast_symbols(content: str, filename: str = "") -> list[CodeSymbol]:
    """
    Extract structural symbols from Python or polyglot source.
    Uses Python's ast for Python files, falling back to regex for polyglot languages.
    """
    symbols: list[CodeSymbol] = []
    if not filename or filename.endswith(".py"):
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    doc = ast.get_docstring(node) or ""
                    symbols.append(
                        CodeSymbol(
                            name=node.name,
                            kind="class",
                            line_number=node.lineno,
                            docstring=doc,
                        )
                    )
                elif isinstance(node, ast.AsyncFunctionDef):
                    doc = ast.get_docstring(node) or ""
                    symbols.append(
                        CodeSymbol(
                            name=node.name,
                            kind="async_function",
                            line_number=node.lineno,
                            docstring=doc,
                        )
                    )
                elif isinstance(node, ast.FunctionDef):
                    doc = ast.get_docstring(node) or ""
                    symbols.append(
                        CodeSymbol(
                            name=node.name,
                            kind="function",
                            line_number=node.lineno,
                            docstring=doc,
                        )
                    )
            if symbols:
                return symbols
        except Exception:
            pass

    # Regex extraction for non-Python or unparseable code
    patterns = [
        # Python
        (r"^\s*(?:class|def|async\s+def)\s+([a-zA-Z_]\w*)", "symbol"),
        # JS/TS
        (r"^\s*(?:export\s+)?(?:default\s+)?(?:class|interface|type)\s+([a-zA-Z_$]\w*)", "type"),
        (r"^\s*(?:export\s+)?(?:async\s+)?function\s+([a-zA-Z_$]\w*)", "function"),
        (
            r"^\s*(?:export\s+)?(?:const|let|var)\s+([a-zA-Z_$]\w*)\s*=\s*(?:async\s*)?\(",
            "function",
        ),
        # Go
        (r"^type\s+([a-zA-Z_]\w*)\s+(?:struct|interface)", "type"),
        (r"^func\s+(?:\([^)]+\)\s+)?([a-zA-Z_]\w*)\s*\(", "function"),
        # Rust
        (r"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+([a-zA-Z_]\w*)", "type"),
        (r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([a-zA-Z_]\w*)", "function"),
        # Java
        (
            r"^\s*(?:public|protected|private)?\s*(?:static\s+)?(?:class|interface|enum|record)\s+([a-zA-Z_]\w*)",
            "type",
        ),
        (
            r"^\s*(?:public|protected|private)?\s*(?:static\s+)?[\w<>\[\]]+\s+([a-zA-Z_]\w*)\s*\([^)]*\)\s*\{?",
            "method",
        ),
    ]

    for line_idx, line in enumerate(content.splitlines(), 1):
        for pat, kind in patterns:
            m = re.search(pat, line)
            if m:
                symbols.append(CodeSymbol(name=m.group(1), kind=kind, line_number=line_idx))
                break

    return symbols


def tokenize_for_search(text: str) -> list[str]:
    """Tokenize text for search — handle snake_case, camelCase, and punctuation."""
    tokens: set[str] = set()
    for w in re.split(r"[^a-zA-Z0-9_]+", text):
        if len(w) > 1:
            tokens.add(w.lower())
            w_no_us = w.replace("_", "")
            if len(w_no_us) > 1:
                tokens.add(w_no_us.lower())
    # camelCase splitting
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    for sub in re.split(r"[^a-zA-Z0-9]+", s1):
        if len(sub) > 1:
            tokens.add(sub.lower())
    return list(tokens)


async def find_relevant_files(
    repo: str,
    query: str,
    exclude: list[str],
    max_results: int = 3,
) -> list[str]:
    """
    Primary interface for finding relevant files in a repository.
    Combines AST symbol extraction + BM25 keyword matching across polyglot files.
    """
    from src.tools._repo_cache import get_local_repo_path

    repo_path = await get_local_repo_path(repo)
    loop = asyncio.get_running_loop()

    def _search() -> list[str]:
        root = Path(repo_path)
        exclude_set = set(exclude)

        corpus: list[tuple[str, list[str], list[str]]] = []  # (rel_path, tokens, symbol_names)
        for fpath in root.rglob("*"):
            if not fpath.is_file():
                continue
            rel = str(fpath.relative_to(root))
            if rel in exclude_set or any(ign in rel for ign in IGNORE_DIRS):
                continue
            suffix = fpath.suffix.lower()
            if suffix not in POLYGLOT_EXTENSIONS and fpath.name not in MANIFEST_FILES:
                continue
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
                symbols = extract_ast_symbols(content, fpath.name)
                sym_names = [s.name for s in symbols]
                docstrings = [s.docstring for s in symbols if s.docstring]

                # Rich token set including file path, symbol names, and docstring terms
                text_to_tokenize = f"{rel} {' '.join(sym_names)} {' '.join(docstrings)}"
                tokens = tokenize_for_search(text_to_tokenize)
                corpus.append((rel, tokens, sym_names))
            except Exception:
                continue

        if not corpus:
            return []

        paths = [c[0] for c in corpus]
        tokenized_corpus = [c[1] for c in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = tokenize_for_search(query)
        scores = bm25.get_scores(query_tokens)

        # Apply structural boost: if query mentions a symbol declared in this file, boost score
        boosted_scores: list[float] = []
        for i, (_, _, sym_names) in enumerate(corpus):
            base_score = scores[i]
            # Check for exact symbol name overlap in query
            symbol_boost = 0.0
            for sym in sym_names:
                if sym.lower() in [q.lower() for q in query_tokens]:
                    symbol_boost += 1.5
            boosted_scores.append(base_score + symbol_boost)

        ranked = sorted(zip(paths, boosted_scores, strict=True), key=lambda x: x[1], reverse=True)

        matching = []
        for path, score in ranked[:max_results]:
            if score > 0:
                matching.append(path)
            else:
                file_tokens = next((c[1] for c in corpus if c[0] == path), [])
                if any(t in file_tokens for t in query_tokens):
                    matching.append(path)
        return matching

    return await loop.run_in_executor(None, _search)


async def search_symbols(
    repo: str,
    symbol_query: str,
) -> list[dict[str, Any]]:
    """Search for code symbols (functions, classes, methods) matching a query."""
    from src.tools._repo_cache import get_local_repo_path

    repo_path = await get_local_repo_path(repo)
    loop = asyncio.get_running_loop()

    def _find_symbols() -> list[dict[str, Any]]:
        root = Path(repo_path)
        results: list[dict[str, Any]] = []
        q_lower = symbol_query.lower()

        for fpath in root.rglob("*"):
            if not fpath.is_file():
                continue
            rel = str(fpath.relative_to(root))
            if any(ign in rel for ign in IGNORE_DIRS):
                continue
            if fpath.suffix.lower() not in POLYGLOT_EXTENSIONS:
                continue
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
                symbols = extract_ast_symbols(content, fpath.name)
                for sym in symbols:
                    name_match = q_lower in sym.name.lower()
                    doc_match = bool(sym.docstring and q_lower in sym.docstring.lower())
                    if name_match or doc_match:
                        results.append(
                            {
                                "file": rel,
                                **sym.to_dict(),
                            }
                        )
            except Exception:
                continue

        return results

    return await loop.run_in_executor(None, _find_symbols)


__all__ = ["find_relevant_files", "search_symbols", "extract_ast_symbols", "tokenize_for_search"]
