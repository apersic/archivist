"""Composition root.

The one place that knows which concrete adapter is used for each port. Nothing
else in the codebase imports a provider, which is what keeps the layers
swappable and the tests fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from archivist.agent.loop import ResearchAgent, plain_rag_answer
from archivist.agent.tools import ToolRegistry, build_default_tools
from archivist.config import Settings
from archivist.generate.answerer import Answerer
from archivist.index.embeddings import build_embedder
from archivist.index.lexical import BM25Index
from archivist.index.vector_store import InMemoryVectorIndex
from archivist.ingest.chunking import chunk_corpus
from archivist.ingest.loaders import load_corpus
from archivist.llm import build_llm
from archivist.ports import LLM, Embedder
from archivist.retrieve.pipeline import RetrievalPipeline
from archivist.retrieve.rerank import build_reranker
from archivist.types import Answer, Chunk


@dataclass
class Archivist:
    settings: Settings
    embedder: Embedder
    llm: LLM
    vector_index: InMemoryVectorIndex
    lexical_index: BM25Index
    pipeline: RetrievalPipeline
    answerer: Answerer
    tools: ToolRegistry
    agent: ResearchAgent

    @property
    def chunk_count(self) -> int:
        return len(self.vector_index)

    def ask(self, question: str, history: list[str] | None = None) -> Answer:
        """Plain RAG: one retrieval, one generation."""
        return plain_rag_answer(question, self.pipeline, self.answerer, history=history)

    def research(self, question: str, history: list[str] | None = None) -> Answer:
        """Agentic RAG: bounded tool-use loop, then one grounded generation."""
        return self.agent.run(question, history=history).answer

    def index_path(self) -> Path:
        return self.settings.index_dir / f"{self.embedder.name}.jsonl"


def build_indexes(
    chunks: list[Chunk], embedder: Embedder
) -> tuple[InMemoryVectorIndex, BM25Index]:
    vector_index = InMemoryVectorIndex(dimensions=embedder.dimensions)
    vector_index.add(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
    lexical_index = BM25Index()
    lexical_index.add(chunks)
    return vector_index, lexical_index


def build(
    settings: Settings | None = None,
    chunks: list[Chunk] | None = None,
    llm: LLM | None = None,
) -> Archivist:
    """Wire the system. Pass `chunks` to skip corpus loading (tests, notebooks)."""
    settings = settings or Settings()
    embedder = build_embedder(
        settings.providers.embedder,
        dimensions=settings.providers.embedding_dimensions,
        model=settings.providers.model,
    )
    if chunks is None:
        chunks = chunk_corpus(load_corpus(settings.corpus_dir), settings.chunking)
    if not chunks:
        raise ValueError(f"no indexable content found in {settings.corpus_dir}")

    resolved_llm = llm or build_llm(settings.providers.llm, settings.providers.model)
    vector_index, lexical_index = build_indexes(chunks, embedder)
    pipeline = RetrievalPipeline(
        embedder=embedder,
        vector_index=vector_index,
        lexical_index=lexical_index,
        reranker=build_reranker(settings.providers.reranker),
        config=settings.retrieval,
    )
    answerer = Answerer(llm=resolved_llm)
    tools = build_default_tools(pipeline, vector_index)
    agent = ResearchAgent(
        llm=resolved_llm,
        pipeline=pipeline,
        tools=tools,
        answerer=answerer,
        config=settings.agent,
    )
    return Archivist(
        settings=settings,
        embedder=embedder,
        llm=resolved_llm,
        vector_index=vector_index,
        lexical_index=lexical_index,
        pipeline=pipeline,
        answerer=answerer,
        tools=tools,
        agent=agent,
    )
