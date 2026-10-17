"""Configuration as data.

Every knob that changes retrieval quality lives here, so an experiment is a
config diff and an eval run rather than an edit to the pipeline. The eval
harness records the config it ran with next to the scores it produced.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    target_tokens: int = 220
    overlap_tokens: int = 40
    min_tokens: int = 24
    respect_headings: bool = True


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    dense_limit: int = 20
    lexical_limit: int = 20
    fusion_k: int = 60
    rerank_limit: int = 6
    context_tokens: int = 1200
    min_score: float = 0.0
    expand_neighbours: bool = True
    query_expansions: int = 1


@dataclass(frozen=True, slots=True)
class AgentConfig:
    max_steps: int = 4
    max_searches: int = 3
    allow_abstain: bool = True


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    embedder: str = "hashing"
    llm: str = "extractive"
    reranker: str = "identity"
    embedding_dimensions: int = 512
    model: str | None = None


@dataclass(frozen=True, slots=True)
class Settings:
    corpus_dir: Path = Path("corpus")
    index_dir: Path = Path(".archivist")
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    providers: ProviderConfig = field(default_factory=ProviderConfig)

    @classmethod
    def from_env(cls, overrides: dict[str, Any] | None = None) -> Settings:
        """Build settings from ARCHIVIST_* environment variables plus overrides."""
        providers = ProviderConfig(
            embedder=os.getenv("ARCHIVIST_EMBEDDER", "hashing"),
            llm=os.getenv("ARCHIVIST_LLM", "extractive"),
            reranker=os.getenv("ARCHIVIST_RERANKER", "identity"),
            embedding_dimensions=int(os.getenv("ARCHIVIST_EMBEDDING_DIMS", "512")),
            model=os.getenv("ARCHIVIST_MODEL"),
        )
        settings = cls(
            corpus_dir=Path(os.getenv("ARCHIVIST_CORPUS", "corpus")),
            index_dir=Path(os.getenv("ARCHIVIST_INDEX", ".archivist")),
            providers=providers,
        )
        return settings.merge(overrides or {})

    def merge(self, overrides: dict[str, Any]) -> Settings:
        """Return a copy with nested dataclass fields replaced by `overrides`."""
        data = self.to_dict()
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                data[key] = {**data[key], **value}
            else:
                data[key] = value
        return Settings(
            corpus_dir=Path(data["corpus_dir"]),
            index_dir=Path(data["index_dir"]),
            chunking=ChunkingConfig(**data["chunking"]),
            retrieval=RetrievalConfig(**data["retrieval"]),
            agent=AgentConfig(**data["agent"]),
            providers=ProviderConfig(**data["providers"]),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["corpus_dir"] = str(self.corpus_dir)
        data["index_dir"] = str(self.index_dir)
        return data

    def fingerprint(self) -> str:
        """Stable identity of the retrieval configuration, for eval bookkeeping."""
        payload = {
            "chunking": asdict(self.chunking),
            "retrieval": asdict(self.retrieval),
            "providers": asdict(self.providers),
        }
        return json.dumps(payload, sort_keys=True)
