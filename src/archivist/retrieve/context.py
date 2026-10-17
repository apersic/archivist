"""Context assembly.

Retrieval returns a ranked list; the generator needs a budgeted, de-duplicated,
citable block of text. This module is where the token budget is enforced and
where each passage gets the marker ([S1], [S2], ...) that the answer must cite.
"""

from __future__ import annotations

from dataclasses import dataclass

from archivist.text import estimate_tokens, tokenize
from archivist.types import Citation, Scored

PASSAGE_SEPARATOR = "\n\n---\n\n"


@dataclass(frozen=True, slots=True)
class AssembledContext:
    text: str
    passages: tuple[Scored, ...]
    citations: tuple[Citation, ...]
    tokens: int

    @property
    def is_empty(self) -> bool:
        return not self.passages

    def citation_for(self, marker: str) -> Citation | None:
        for citation in self.citations:
            if citation.marker == marker:
                return citation
        return None


def _jaccard(left: str, right: str) -> float:
    a, b = set(tokenize(left)), set(tokenize(right))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def assemble(
    candidates: list[Scored],
    budget_tokens: int,
    near_duplicate_threshold: float = 0.85,
) -> AssembledContext:
    """Pack the highest-ranked, non-redundant passages into the token budget."""
    kept: list[Scored] = []
    citations: list[Citation] = []
    blocks: list[str] = []
    used = 0

    for candidate in candidates:
        if any(
            _jaccard(candidate.chunk.text, existing.chunk.text) >= near_duplicate_threshold
            for existing in kept
        ):
            continue
        tokens = candidate.chunk.token_estimate or estimate_tokens(candidate.chunk.text)
        if used + tokens > budget_tokens and kept:
            continue
        marker = f"S{len(kept) + 1}"
        kept.append(candidate)
        citations.append(
            Citation(
                marker=marker,
                chunk_id=candidate.chunk.chunk_id,
                location=candidate.chunk.location(),
            )
        )
        blocks.append(f"[{marker}] {candidate.chunk.location()}\n{candidate.chunk.text}")
        used += tokens
        if used >= budget_tokens:
            break

    return AssembledContext(
        text=PASSAGE_SEPARATOR.join(blocks),
        passages=tuple(kept),
        citations=tuple(citations),
        tokens=used,
    )
