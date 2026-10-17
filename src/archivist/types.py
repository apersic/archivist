"""Data contracts shared by every layer.

These types are the only thing the layers agree on. Anything that is not here
(an embedding model, a vector database, an LLM) is an implementation detail
hidden behind a port in `archivist.ports`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    """A source file after loading and normalisation to text."""

    doc_id: str
    text: str
    source: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit of a document.

    `ordinal` and `section` exist so that a retrieved chunk can be expanded to
    its neighbours ("small-to-big" retrieval) and cited with a human-readable
    location instead of an opaque id.
    """

    chunk_id: str
    doc_id: str
    text: str
    ordinal: int
    source: str
    title: str | None = None
    section: str | None = None
    token_estimate: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def location(self) -> str:
        label = self.title or self.source
        return f"{label} / {self.section}" if self.section else label


@dataclass(frozen=True, slots=True)
class Scored:
    """A chunk with the score of the retriever that produced it."""

    chunk: Chunk
    score: float
    retriever: str = "unknown"


@dataclass(frozen=True, slots=True)
class Citation:
    marker: str
    chunk_id: str
    location: str


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Answer:
    question: str
    text: str
    status: AnswerStatus
    citations: tuple[Citation, ...] = ()
    contexts: tuple[Scored, ...] = ()
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def cited_chunk_ids(self) -> tuple[str, ...]:
        return tuple(c.chunk_id for c in self.citations)


@dataclass(frozen=True, slots=True)
class LLMMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    usage: dict[str, int] = field(default_factory=dict)
    tool_call: ToolCall | None = None


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    name: str
    content: str
    chunks: tuple[Scored, ...] = ()
