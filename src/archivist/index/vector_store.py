"""Vector index adapters.

`InMemoryVectorIndex` does exact cosine search in pure Python. Exact search is
the right default: it removes recall loss from the ANN layer while you are
still debugging chunking and prompts, and a corpus of a few thousand chunks
does not need a database. The persistence format is JSONL so an index is
diffable and inspectable by hand.

When the corpus outgrows memory, implement `ports.VectorIndex` over pgvector
or Qdrant; `recall@k` from the eval harness is what tells you whether the ANN
parameters you chose cost you anything.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from archivist.types import Chunk, Scored


def cosine(left: list[float], right: list[float]) -> float:
    """Dot product; vectors from `Embedder` implementations are L2-normalised."""
    return sum(a * b for a, b in zip(left, right, strict=True))


class InMemoryVectorIndex:
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}
        self._by_doc: dict[str, list[str]] = {}

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"vector for {chunk.chunk_id} has {len(vector)} dims, "
                    f"index expects {self.dimensions}"
                )
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = vector
            self._by_doc.setdefault(chunk.doc_id, []).append(chunk.chunk_id)

    def search(self, vector: list[float], limit: int) -> list[Scored]:
        scores = [
            Scored(chunk=self._chunks[chunk_id], score=cosine(vector, candidate), retriever="dense")
            for chunk_id, candidate in self._vectors.items()
        ]
        scores.sort(key=lambda item: item.score, reverse=True)
        return scores[:limit]

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def neighbours(self, chunk_id: str, window: int = 1) -> list[Chunk]:
        """Adjacent chunks of the same document, for small-to-big expansion."""
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            return []
        siblings = sorted(
            (self._chunks[cid] for cid in self._by_doc.get(chunk.doc_id, ())),
            key=lambda c: c.ordinal,
        )
        return [
            sibling
            for sibling in siblings
            if sibling.chunk_id != chunk_id
            and abs(sibling.ordinal - chunk.ordinal) <= window
        ]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def __len__(self) -> int:
        return len(self._chunks)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"dimensions": self.dimensions}) + "\n")
            for chunk_id, chunk in self._chunks.items():
                record = {"chunk": asdict(chunk), "vector": self._vectors[chunk_id]}
                handle.write(json.dumps(record) + "\n")

    @classmethod
    def load(cls, path: Path) -> InMemoryVectorIndex:
        with path.open(encoding="utf-8") as handle:
            header = json.loads(handle.readline())
            index = cls(dimensions=int(header["dimensions"]))
            chunks: list[Chunk] = []
            vectors: list[list[float]] = []
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                chunks.append(Chunk(**record["chunk"]))
                vectors.append([float(value) for value in record["vector"]])
        index.add(chunks, vectors)
        return index
