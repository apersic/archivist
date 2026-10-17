"""LLM adapters and their factory."""

from __future__ import annotations

from archivist.llm.extractive import ExtractiveLLM, ScriptedLLM
from archivist.llm.openai_client import OpenAIChatLLM
from archivist.ports import LLM

__all__ = ["ExtractiveLLM", "OpenAIChatLLM", "ScriptedLLM", "build_llm"]


def build_llm(name: str, model: str | None = None) -> LLM:
    match name:
        case "extractive":
            return ExtractiveLLM()
        case "openai":
            return OpenAIChatLLM(model=model or "gpt-4o-mini")
        case _:
            raise ValueError(f"unknown llm: {name!r} (expected 'extractive' or 'openai')")
