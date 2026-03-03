"""
Example 1: Deterministic Quoting with Sentence-Level Indexing

This package demonstrates the difference between:
- WRONG way: Free-form quote generation (LLM writes quotes directly)
- RIGHT way: Index-based selection (LLM selects sentence indices)

The key insight: When you need exact, verifiable quotes from source text,
never let the LLM generate the text. Instead, let it choose indices,
then retrieve the exact text deterministically.
"""

from .wrong_way import extract_quotes_wrong_way
from .right_way import extract_quotes_right_way, index_sentences, IndexedSentences

__all__ = [
    "extract_quotes_wrong_way",
    "extract_quotes_right_way",
    "index_sentences",
    "IndexedSentences",
]