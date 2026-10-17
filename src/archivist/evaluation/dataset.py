"""Golden set handling.

A case is a question plus what a correct system does with it: which chunks are
relevant, which facts the answer must contain, and whether the right behaviour
is to abstain. Unanswerable cases are first-class - a RAG system that cannot
refuse is not safe to put in front of users.

`relevant_sources` is matched against chunk *sources* rather than chunk ids so
the golden set survives a re-chunking. That matters: otherwise every chunking
experiment invalidates your labels.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from archivist.text import tokenize
from archivist.types import Chunk


@dataclass(frozen=True, slots=True)
class EvalCase:
    question: str
    relevant_sources: tuple[str, ...] = ()
    must_contain: tuple[str, ...] = ()
    unanswerable: bool = False
    tags: tuple[str, ...] = ()
    history: tuple[str, ...] = field(default=())

    def relevant_chunk_ids(self, chunks: list[Chunk]) -> set[str]:
        return {chunk.chunk_id for chunk in chunks if chunk.source in self.relevant_sources}


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("//"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
            cases.append(
                EvalCase(
                    question=record["question"],
                    relevant_sources=tuple(record.get("relevant_sources", ())),
                    must_contain=tuple(record.get("must_contain", ())),
                    unanswerable=bool(record.get("unanswerable", False)),
                    tags=tuple(record.get("tags", ())),
                    history=tuple(record.get("history", ())),
                )
            )
    return cases


def save_cases(cases: list[EvalCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")


def propose_cases(chunks: list[Chunk], per_source: int = 1) -> list[EvalCase]:
    """Seed a golden set from the corpus: draft questions for a human to fix.

    Synthetic questions are a starting point, never the ground truth. They are
    biased toward what the corpus says loudly; the questions that matter are the
    ones users actually ask. Treat this as a labelling accelerator.
    """
    drafts: list[EvalCase] = []
    by_source: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source, []).append(chunk)

    for source, source_chunks in sorted(by_source.items()):
        ranked = sorted(source_chunks, key=lambda c: c.token_estimate, reverse=True)
        for chunk in ranked[:per_source]:
            terms = [token for token in tokenize(chunk.text) if len(token) > 5][:4]
            topic = chunk.section or chunk.title or source
            drafts.append(
                EvalCase(
                    question=f"What does the documentation say about {topic}?",
                    relevant_sources=(source,),
                    must_contain=tuple(terms[:2]),
                    tags=("synthetic", "draft"),
                )
            )
    return drafts
