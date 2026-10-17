"""BM25 lexical index.

Dense retrieval fails in a predictable way: it cannot reliably find a rare,
exact string (an error code, a config key, a surname). BM25 is ~60 lines and
fixes exactly that failure, which is why hybrid retrieval beats either half.
"""

from __future__ import annotations

import math
from collections import Counter

from archivist.text import STOPWORDS, tokenize
from archivist.types import Chunk, Scored


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75, drop_stopwords: bool = True) -> None:
        self.k1 = k1
        self.b = b
        self.drop_stopwords = drop_stopwords
        self._chunks: list[Chunk] = []
        self._term_frequencies: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._document_frequencies: Counter[str] = Counter()

    def _terms(self, text: str) -> list[str]:
        tokens = tokenize(text)
        if not self.drop_stopwords:
            return tokens
        return [token for token in tokens if token not in STOPWORDS]

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            terms = self._terms(chunk.text)
            frequencies = Counter(terms)
            self._chunks.append(chunk)
            self._term_frequencies.append(frequencies)
            self._lengths.append(len(terms))
            self._document_frequencies.update(frequencies.keys())

    @property
    def average_length(self) -> float:
        return sum(self._lengths) / len(self._lengths) if self._lengths else 0.0

    def _idf(self, term: str) -> float:
        total = len(self._chunks)
        frequency = self._document_frequencies.get(term, 0)
        if frequency == 0:
            return 0.0
        return math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))

    def search(self, query: str, limit: int) -> list[Scored]:
        query_terms = self._terms(query)
        if not query_terms or not self._chunks:
            return []
        average = self.average_length or 1.0
        results: list[Scored] = []

        for position, chunk in enumerate(self._chunks):
            frequencies = self._term_frequencies[position]
            length = self._lengths[position] or 1
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                numerator = frequency * (self.k1 + 1)
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / average)
                score += self._idf(term) * numerator / denominator
            if score > 0.0:
                results.append(Scored(chunk=chunk, score=score, retriever="lexical"))

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    def __len__(self) -> int:
        return len(self._chunks)
