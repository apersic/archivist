"""Rerankers.

The retrieval stage optimises recall over a wide candidate set; the reranker
optimises precision over a narrow one. Keeping them separate is what lets you
retrieve 40 candidates cheaply and spend the expensive model on only those.

`LexicalOverlapReranker` is a dependency-free stand-in that scores candidates
on query-term coverage and proximity. `CrossEncoderReranker` is the real thing.
"""

from __future__ import annotations

from typing import Any

from archivist.ports import Reranker
from archivist.text import tokenize
from archivist.types import Scored


class IdentityReranker:
    """Baseline: keep fusion order. Always eval against this."""

    @property
    def name(self) -> str:
        return "identity"

    def rerank(
        self,
        query: str,  # noqa: ARG002 - part of the Reranker port
        candidates: list[Scored],
        limit: int,
    ) -> list[Scored]:
        return candidates[:limit]


class LexicalOverlapReranker:
    """Coverage of query terms, plus a bonus when matched terms appear close together."""

    def __init__(self, coverage_weight: float = 1.0, proximity_weight: float = 0.35) -> None:
        self.coverage_weight = coverage_weight
        self.proximity_weight = proximity_weight

    @property
    def name(self) -> str:
        return "lexical-overlap"

    def _score(self, query_terms: list[str], text: str) -> float:
        tokens = tokenize(text)
        if not tokens or not query_terms:
            return 0.0
        positions = {
            term: [index for index, token in enumerate(tokens) if token == term]
            for term in query_terms
        }
        matched = [term for term, hits in positions.items() if hits]
        coverage = len(matched) / len(query_terms)

        proximity = 0.0
        if len(matched) > 1:
            firsts = sorted(positions[term][0] for term in matched)
            spread = firsts[-1] - firsts[0] + 1
            proximity = len(matched) / spread
        return self.coverage_weight * coverage + self.proximity_weight * proximity

    def rerank(self, query: str, candidates: list[Scored], limit: int) -> list[Scored]:
        query_terms = list(dict.fromkeys(tokenize(query)))
        rescored = [
            Scored(
                chunk=candidate.chunk,
                # Fusion score is kept as a small tie-breaker so a candidate that
                # both retrievers liked is not discarded on a lexical technicality.
                score=self._score(query_terms, candidate.chunk.text) + 0.1 * candidate.score,
                retriever=f"{candidate.retriever}|reranked",
            )
            for candidate in candidates
        ]
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:limit]


class CrossEncoderReranker:
    """Adapter for a sentence-transformers cross-encoder (`local-embeddings` extra)."""

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        encoder: Any = None,
    ) -> None:
        self._model = model
        self._encoder = encoder

    @property
    def name(self) -> str:
        return self._model

    def rerank(self, query: str, candidates: list[Scored], limit: int) -> list[Scored]:
        if self._encoder is None:
            raise RuntimeError("install the 'local-embeddings' extra and pass an encoder")
        pairs = [(query, candidate.chunk.text) for candidate in candidates]
        scores = self._encoder.predict(pairs)
        rescored = [
            Scored(chunk=candidate.chunk, score=float(score), retriever="cross-encoder")
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:limit]


def build_reranker(name: str) -> Reranker:
    match name:
        case "identity" | "none":
            return IdentityReranker()
        case "lexical-overlap":
            return LexicalOverlapReranker()
        case "cross-encoder":
            return CrossEncoderReranker()
        case _:
            raise ValueError(f"unknown reranker: {name!r}")
