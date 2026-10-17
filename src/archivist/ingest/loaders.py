"""Corpus loading.

Loaders are intentionally dumb: find files, normalise to UTF-8 text, record
where the text came from. Format-specific extraction (PDF, HTML, notebooks)
plugs in as another entry in `LOADERS` and nothing downstream changes.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from archivist.types import Document

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}


def _stable_id(prefix: str, payload: str) -> str:
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"


def _title_from_markdown(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def load_text_file(path: Path, root: Path) -> Document:
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\r\n?", "\n", raw).strip()
    # Eval labels and citations use POSIX paths so they match on every OS.
    relative = path.relative_to(root) if path.is_relative_to(root) else path
    source = relative.as_posix()
    return Document(
        doc_id=_stable_id("doc", source),
        text=text,
        source=source,
        title=_title_from_markdown(text, path.stem.replace("-", " ").replace("_", " ")),
        metadata={"bytes": len(raw), "suffix": path.suffix},
    )


LOADERS: dict[str, Callable[[Path, Path], Document]] = dict.fromkeys(
    TEXT_SUFFIXES, load_text_file
)


def iter_corpus_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in LOADERS:
            yield path


def load_corpus(root: Path) -> list[Document]:
    """Load every supported file under `root`, skipping unreadable ones."""
    if not root.exists():
        raise FileNotFoundError(f"corpus directory not found: {root}")
    return [load_text_file(path, root) for path in iter_corpus_files(root)]


def load_documents(paths: Iterable[Path], root: Path) -> list[Document]:
    return [load_text_file(path, root) for path in paths]
