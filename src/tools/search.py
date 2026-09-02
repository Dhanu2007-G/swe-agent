"""
src/tools/search.py — Semantic + keyword search over repo code.
Delegates to filesystem.py's find_relevant_files for BM25 search.
"""
from src.tools.filesystem import find_relevant_files

__all__ = ["find_relevant_files"]
