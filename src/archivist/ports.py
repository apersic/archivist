"""Ports: the seams where a toy implementation is swapped for a real one.

Each protocol is deliberately small. The default adapters are dependency-free
so the system is testable offline; the production adapters (OpenAI embeddings,
pgvector, a cross-encoder reranker) implement the same protocol and are chosen
by configuration, never by an `if` inside a pipeline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from archivist.types import Chunk, LLMMessage, LLMResponse, Scored


@runtime_checkable
class Embedder(Protocol):
    """Turns text into dense vectors."""

    @property
    def name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class VectorIndex(Protocol):
    """Approximate or exact nearest-neighbour search over chunk embeddings."""

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(self, vector: list[float], limit: int) -> list[Scored]: ...

    def get(self, chunk_id: str) -> Chunk | None: ...

    def __len__(self) -> int: ...


@runtime_checkable
class LexicalIndex(Protocol):
    """Keyword search. Catches the exact identifiers that embeddings blur."""

    def add(self, chunks: list[Chunk]) -> None: ...

    def search(self, query: str, limit: int) -> list[Scored]: ...


@runtime_checkable
class Reranker(Protocol):
    """Re-scores a candidate set with a slower, more accurate model."""

    @property
    def name(self) -> str: ...

    def rerank(self, query: str, candidates: list[Scored], limit: int) -> list[Scored]: ...


@runtime_checkable
class LLM(Protocol):
    """Text generation. Tool calling is expressed through `LLMResponse.tool_call`."""

    @property
    def name(self) -> str: ...

    def complete(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, object]] | None = None,
    ) -> LLMResponse: ...
