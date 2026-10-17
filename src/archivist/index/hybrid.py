"""Fusion of several ranked lists.

Reciprocal rank fusion is the default because it only needs ranks, not scores:
cosine similarities and BM25 scores are not on a comparable scale, and
normalising them introduces a tuning parameter per corpus. RRF introduces one
constant (`k`) that is famously insensitive.
"""

from __future__ import annotations

from collections.abc import Sequence

from archivist.types import Scored


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Scored]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[Scored]:
    """Combine ranked lists into one, scoring each chunk by sum(w / (k + rank))."""
    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError("weights must match the number of ranked lists")

    fused: dict[str, float] = {}
    best: dict[str, Scored] = {}
    sources: dict[str, set[str]] = {}

    for list_position, results in enumerate(ranked_lists):
        weight = weights[list_position] if weights else 1.0
        for rank, scored in enumerate(results, start=1):
            chunk_id = scored.chunk.chunk_id
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (k + rank)
            sources.setdefault(chunk_id, set()).add(scored.retriever)
            if chunk_id not in best or rank == 1:
                best[chunk_id] = scored

    merged = [
        Scored(
            chunk=best[chunk_id].chunk,
            score=score,
            retriever="+".join(sorted(sources[chunk_id])),
        )
        for chunk_id, score in fused.items()
    ]
    merged.sort(key=lambda item: (item.score, item.chunk.chunk_id), reverse=True)
    return merged
