"""Archivist: an agentic RAG system over a private document corpus."""

from __future__ import annotations

from archivist.app import Archivist, build
from archivist.config import Settings
from archivist.types import Answer, AnswerStatus, Chunk, Citation, Document

__all__ = [
    "Answer",
    "AnswerStatus",
    "Archivist",
    "Chunk",
    "Citation",
    "Document",
    "Settings",
    "build",
]
__version__ = "0.1.0"
