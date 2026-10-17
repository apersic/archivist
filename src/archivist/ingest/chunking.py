"""Structure-aware chunking.

Chunking is the single highest-leverage knob in a RAG system: it decides what
"one fact" means. The strategy here is two-pass.

1. Split on Markdown headings so a chunk never straddles two topics, and record
   the heading path as the chunk's section for citations.
2. Window each section to a token budget with overlap, breaking on paragraph
   then sentence boundaries so a chunk rarely ends mid-thought.

Token counts are estimated, not tokenised, to keep the core dependency-free.
Swap in `tiktoken` behind `estimate_tokens` when exact budgeting matters.
"""

from __future__ import annotations

import re

from archivist.config import ChunkingConfig
from archivist.text import estimate_tokens
from archivist.types import Chunk, Document

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _heading_path(stack: list[tuple[int, str]]) -> str | None:
    return " > ".join(title for _, title in stack) if stack else None


def split_sections(text: str, respect_headings: bool) -> list[tuple[str | None, str]]:
    """Split a document into (section path, body) pairs."""
    if not respect_headings:
        return [(None, text)]

    sections: list[tuple[str | None, str]] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []

    def flush() -> None:
        joined = "\n".join(body).strip()
        if joined:
            sections.append((_heading_path(stack), joined))
        body.clear()

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2).strip()))
            continue
        body.append(line)
    flush()
    return sections or [(None, text)]


def _split_units(body: str) -> list[str]:
    """Paragraphs, further split into sentences when a paragraph is huge."""
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", body):
        cleaned = paragraph.strip()
        if not cleaned:
            continue
        if estimate_tokens(cleaned) <= 400:
            units.append(cleaned)
            continue
        units.extend(part.strip() for part in _SENTENCE_END.split(cleaned) if part.strip())
    return units


def _window(units: list[str], config: ChunkingConfig) -> list[str]:
    """Greedily pack units up to the target budget, then overlap backwards."""
    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = estimate_tokens(unit)
        if current and current_tokens + unit_tokens > config.target_tokens:
            windows.append("\n\n".join(current))
            carry: list[str] = []
            carry_tokens = 0
            for previous in reversed(current):
                previous_tokens = estimate_tokens(previous)
                if carry_tokens + previous_tokens > config.overlap_tokens:
                    break
                carry.insert(0, previous)
                carry_tokens += previous_tokens
            current = carry
            current_tokens = carry_tokens
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        windows.append("\n\n".join(current))
    return windows


def _relative_section(section: str | None, title: str | None) -> str | None:
    """Drop the document's own H1 from the heading path so labels do not repeat."""
    if section is None or title is None:
        return section
    if section == title:
        return None
    return section.removeprefix(f"{title} > ") or None


def chunk_document(document: Document, config: ChunkingConfig | None = None) -> list[Chunk]:
    config = config or ChunkingConfig()
    chunks: list[Chunk] = []
    ordinal = 0

    for raw_section, body in split_sections(document.text, config.respect_headings):
        section = _relative_section(raw_section, document.title)
        units = _split_units(body)
        for window in _window(units, config):
            tokens = estimate_tokens(window)
            if tokens < config.min_tokens and chunks and chunks[-1].section == section:
                # Too small to retrieve on its own: fold it into the previous chunk.
                previous = chunks.pop()
                window = f"{previous.text}\n\n{window}"
                tokens = estimate_tokens(window)
                ordinal = previous.ordinal
            header = f"{document.title or document.source}"
            prefix = f"{header} — {section}\n\n" if section else f"{header}\n\n"
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}:{ordinal:04d}",
                    doc_id=document.doc_id,
                    # The contextual prefix is part of the indexed text: it makes a
                    # chunk retrievable by terms that only appear in its heading.
                    text=prefix + window,
                    ordinal=ordinal,
                    source=document.source,
                    title=document.title,
                    section=section,
                    token_estimate=tokens,
                    metadata=dict(document.metadata),
                )
            )
            ordinal += 1

    return chunks


def chunk_corpus(documents: list[Document], config: ChunkingConfig | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, config))
    return chunks
