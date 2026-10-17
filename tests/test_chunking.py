from __future__ import annotations

from archivist.config import ChunkingConfig
from archivist.ingest.chunking import chunk_document, estimate_tokens, split_sections
from archivist.types import Document

SAMPLE = """# Service Guide

Intro paragraph about the service.

## Limits

The per-key limit is 2000 events per second.

### Burst

Bursts of 3x the limit are tolerated for 10 seconds.

## Errors

Error `SVC-429` means the caller was rate limited.
"""


def _document(text: str = SAMPLE) -> Document:
    return Document(doc_id="doc-1", text=text, source="guide.md", title="Service Guide")


def test_sections_carry_their_heading_path() -> None:
    sections = split_sections(SAMPLE, respect_headings=True)
    paths = [path for path, _ in sections]
    assert "Service Guide > Limits > Burst" in paths


def test_chunks_record_section_relative_to_the_document_title() -> None:
    chunks = chunk_document(_document())
    sections = {chunk.section for chunk in chunks}
    assert "Limits" in sections
    assert "Limits > Burst" in sections
    # The document's own H1 is not repeated inside the section path.
    assert not any(section and section.startswith("Service Guide") for section in sections)


def test_chunk_text_is_prefixed_with_its_location() -> None:
    chunks = chunk_document(_document())
    errors = next(chunk for chunk in chunks if chunk.section == "Errors")
    assert errors.text.startswith("Service Guide — Errors")
    assert "SVC-429" in errors.text


def test_chunks_respect_the_token_budget_with_overlap() -> None:
    body = "\n\n".join(f"Paragraph {index} with a few words in it." for index in range(60))
    config = ChunkingConfig(target_tokens=60, overlap_tokens=20)
    chunks = chunk_document(_document(f"# Big\n\n{body}"), config)

    assert len(chunks) > 1
    assert all(chunk.token_estimate <= config.target_tokens * 1.5 for chunk in chunks)
    # Overlap means consecutive chunks share text.
    first_tail = chunks[0].text.splitlines()[-1]
    assert first_tail in chunks[1].text


def test_ordinals_are_dense_and_ordered() -> None:
    chunks = chunk_document(_document())
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_token_estimate_is_monotonic() -> None:
    assert estimate_tokens("one two three") < estimate_tokens("one two three four five")
