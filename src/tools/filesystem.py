"""
src/tools/filesystem.py — Repo file access + context loading.
src/tools/search.py is merged here for simplicity.
Uses tree-sitter for AST-aware context extraction.
"""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any

import structlog
import tiktoken

from src.config import get_settings

log = structlog.get_logger(__name__)

# Languages we understand structurally
LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".rs": "rust",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".sh": "bash",
}

IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".env", "dist", "build", ".pytest_cache", ".mypy_cache",
    "*.egg-info",
}

IGNORE_FILES = {
    ".DS_Store", "Thumbs.db", "*.pyc", "*.pyo",
    "*.lock", "package-lock.json",
}

_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


async def list_repo_tree(repo_full_name: str, max_depth: int = 3) -> str:
    """
    Return a tree-formatted string of the repo structure.
    Runs in a thread since it's disk I/O.
    """
    from src.tools._repo_cache import get_local_repo_path

    repo_path = await get_local_repo_path(repo_full_name)
    loop = asyncio.get_running_loop()

    def _build_tree() -> str:
        lines: list[str] = []
        root = Path(repo_path)

        def _walk(path: Path, depth: int, prefix: str = "") -> None:
            if depth > max_depth:
                return
            try:
                entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
            except PermissionError:
                return

            for i, entry in enumerate(entries):
                if entry.name in IGNORE_DIRS or entry.name.startswith("."):
                    continue
                connector = "└── " if i == len(entries) - 1 else "├── "
                lines.append(f"{prefix}{connector}{entry.name}")
                if entry.is_dir():
                    extension = "    " if i == len(entries) - 1 else "│   "
                    _walk(entry, depth + 1, prefix + extension)

        _walk(root, 0)
        return "\n".join(lines[:200])  # cap at 200 lines

    return await loop.run_in_executor(None, _build_tree)


async def list_test_files(repo_full_name: str) -> list[str]:
    """Return relative paths of all test files."""
    from src.tools._repo_cache import get_local_repo_path

    repo_path = await get_local_repo_path(repo_full_name)
    loop = asyncio.get_running_loop()

    def _find() -> list[str]:
        root = Path(repo_path)
        results = []
        for pattern in ["test_*.py", "*_test.py", "tests/**/*.py"]:
            for path in root.glob(pattern):
                rel = str(path.relative_to(root))
                if not any(ign in rel for ign in IGNORE_DIRS):
                    results.append(rel)
        return sorted(set(results))

    return await loop.run_in_executor(None, _find)


async def load_file_contexts(
    repo: str,
    paths: list[str],
    max_tokens_per_file: int = 4000,
) -> list[dict[str, Any]]:
    """
    Load file contents, detect language, count tokens, and truncate if needed.
    Returns serialized dicts (not Pydantic models) for JSON-safe storage in state.
    """
    from src.tools._repo_cache import get_local_repo_path

    repo_path = await get_local_repo_path(repo)
    loop = asyncio.get_running_loop()

    def _load_one(rel_path: str) -> dict[str, Any] | None:
        full_path = Path(repo_path) / rel_path
        if not full_path.exists() or not full_path.is_file():
            log.warning("filesystem.file_not_found", path=rel_path)
            return None

        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            log.warning("filesystem.read_error", path=rel_path, error=str(e))
            return None

        suffix = full_path.suffix.lower()
        language = LANGUAGE_MAP.get(suffix, "text")
        token_count = count_tokens(content)

        # Truncate if over limit
        if token_count > max_tokens_per_file:
            ratio = max_tokens_per_file / token_count
            cutoff = int(len(content) * ratio)
            content = content[:cutoff] + "\n... [TRUNCATED — see full file in repo]"
            token_count = max_tokens_per_file

        return {
            "path": rel_path,
            "content": content,
            "language": language,
            "size_bytes": full_path.stat().st_size,
            "token_count": token_count,
        }

    tasks = [
        loop.run_in_executor(None, _load_one, p)
        for p in paths
    ]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


async def find_relevant_files(
    repo: str,
    query: str,
    exclude: list[str],
    max_results: int = 3,
) -> list[str]:
    """
    BM25 + keyword search to find files relevant to the task query.
    Uses function/class name extraction for better signal.
    """
    from rank_bm25 import BM25Okapi
    from src.tools._repo_cache import get_local_repo_path

    repo_path = await get_local_repo_path(repo)
    loop = asyncio.get_running_loop()

    def _search() -> list[str]:
        root = Path(repo_path)
        exclude_set = set(exclude)

        # Collect all Python files with their extracted symbols
        corpus: list[tuple[str, list[str]]] = []
        for py_file in root.rglob("*.py"):
            rel = str(py_file.relative_to(root))
            if rel in exclude_set:
                continue
            if any(ign in rel for ign in IGNORE_DIRS):
                continue
            try:
                content = py_file.read_text(encoding="utf-8", errors="replace")
                symbols = _extract_symbols(content)
                tokens = _tokenize_for_bm25(rel + " " + " ".join(symbols))
                corpus.append((rel, tokens))
            except Exception:
                continue

        if not corpus:
            return []

        paths = [c[0] for c in corpus]
        tokenized_corpus = [c[1] for c in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = _tokenize_for_bm25(query)
        scores = bm25.get_scores(query_tokens)
        ranked = sorted(zip(paths, scores), key=lambda x: x[1], reverse=True)

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


def _extract_symbols(content: str) -> list[str]:
    """Extract function/class names without a full AST parser (fast)."""
    symbols = []
    for match in re.finditer(
        r"^\s*(?:class|def|async def)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        content, re.MULTILINE
    ):
        symbols.append(match.group(1))
    return symbols


def _tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text for BM25 — split on non-alphanumeric, underscores, camelCase, lowercase."""
    tokens: set[str] = set()
    for w in re.split(r"[^a-zA-Z0-9_]+", text):
        if len(w) > 1:
            tokens.add(w.lower())
            w_no_us = w.replace("_", "")
            if len(w_no_us) > 1:
                tokens.add(w_no_us.lower())
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    for sub in re.split(r"[^a-zA-Z0-9]+", s1):
        if len(sub) > 1:
            tokens.add(sub.lower())
    return list(tokens)
