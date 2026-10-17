"""Shared text utilities.

Tokenisation and stopwords are used by the embedder, the lexical index, the
reranker and the metrics. They live here so all four agree: if the lexical
index and the metrics disagree about what a word is, the numbers stop meaning
anything.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9_]+")

# Underscores are kept inside tokens on purpose: `events_raw` is one identifier,
# not two words, and splitting it destroys the most useful term in the corpus.
STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "does",
        "for", "from", "has", "have", "how", "if", "in", "into", "is", "it",
        "its", "of", "on", "or", "that", "the", "their", "then", "there",
        "these", "this", "to", "was", "were", "what", "when", "where", "which",
        "who", "why", "will", "with",
    }
)


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def content_terms(text: str, min_length: int = 3) -> set[str]:
    """Tokens that carry meaning: no stopwords, no very short tokens."""
    return {
        token
        for token in tokenize(text)
        if token not in STOPWORDS and len(token) >= min_length
    }


def estimate_tokens(text: str) -> int:
    """Rough token count: words plus a surcharge for punctuation-heavy text.

    Deliberately an estimate. Swap in `tiktoken` here when exact budgeting
    against a specific model's tokeniser starts to matter; nothing else in the
    codebase counts tokens.
    """
    words = len(text.split())
    return max(1, int(words * 1.3)) if words else 0
