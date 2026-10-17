"""The retrieval pipeline.

    rewrite -> expand -> (dense || lexical) -> fuse -> expand neighbours -> rerank -> assemble

Every stage is optional and configured, and every stage opens a trace span, so
a bad answer can be attributed to the stage that caused it. The pipeline owns
no models: it is handed ports.
"""

from __future__ import annotations

from dataclasses import dataclass

from archivist.config import RetrievalConfig
from archivist.index.hybrid import reciprocal_rank_fusion
from archivist.index.lexical import BM25Index
from archivist.index.vector_store import InMemoryVectorIndex
from archivist.obs.trace import Trace
from archivist.ports import LLM, Embedder, Reranker
from archivist.retrieve.context import AssembledContext, assemble
from archivist.retrieve.query import expand_question, rewrite_question
from archivist.types import Scored


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    question: str
    standalone_question: str
    candidates: tuple[Scored, ...]
    context: AssembledContext

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(scored.chunk.chunk_id for scored in self.candidates)


class RetrievalPipeline:
    def __init__(
        self,
        embedder: Embedder,
        vector_index: InMemoryVectorIndex,
        lexical_index: BM25Index,
        reranker: Reranker,
        config: RetrievalConfig | None = None,
        query_llm: LLM | None = None,
    ) -> None:
        self.embedder = embedder
        self.vector_index = vector_index
        self.lexical_index = lexical_index
        self.reranker = reranker
        self.config = config or RetrievalConfig()
        self.query_llm = query_llm

    def _dense(self, questions: list[str], trace: Trace) -> list[list[Scored]]:
        results: list[list[Scored]] = []
        with trace.span("retrieve.dense", queries=len(questions)) as span:
            for question in questions:
                vector = self.embedder.embed_query(question)
                results.append(self.vector_index.search(vector, self.config.dense_limit))
            span.attributes["hits"] = sum(len(hits) for hits in results)
        return results

    def _lexical(self, questions: list[str], trace: Trace) -> list[list[Scored]]:
        results: list[list[Scored]] = []
        with trace.span("retrieve.lexical", queries=len(questions)) as span:
            for question in questions:
                results.append(self.lexical_index.search(question, self.config.lexical_limit))
            span.attributes["hits"] = sum(len(hits) for hits in results)
        return results

    def _expand_neighbours(self, fused: list[Scored]) -> list[Scored]:
        """Pull in adjacent chunks of the top hits so a split answer stays whole."""
        seen = {scored.chunk.chunk_id for scored in fused}
        additions: list[Scored] = []
        for scored in fused[: self.config.rerank_limit]:
            for neighbour in self.vector_index.neighbours(scored.chunk.chunk_id):
                if neighbour.chunk_id in seen:
                    continue
                seen.add(neighbour.chunk_id)
                additions.append(
                    Scored(chunk=neighbour, score=scored.score * 0.5, retriever="neighbour")
                )
        return fused + additions

    def run(
        self,
        question: str,
        history: list[str] | None = None,
        trace: Trace | None = None,
    ) -> RetrievalResult:
        trace = trace or Trace()
        with trace.span("retrieve", question=question[:80]):
            with trace.span("retrieve.rewrite"):
                standalone = rewrite_question(question, history or [], self.query_llm)
            with trace.span("retrieve.expand"):
                questions = expand_question(
                    standalone, self.config.query_expansions, self.query_llm
                )

            ranked_lists = self._dense(questions, trace) + self._lexical(questions, trace)

            with trace.span("retrieve.fuse", lists=len(ranked_lists)) as span:
                fused = reciprocal_rank_fusion(ranked_lists, k=self.config.fusion_k)
                fused = [scored for scored in fused if scored.score >= self.config.min_score]
                span.attributes["candidates"] = len(fused)

            if self.config.expand_neighbours:
                fused = self._expand_neighbours(fused)

            with trace.span("retrieve.rerank", reranker=self.reranker.name) as span:
                reranked = self.reranker.rerank(standalone, fused, self.config.rerank_limit)
                span.attributes["kept"] = len(reranked)

            with trace.span("retrieve.assemble", budget=self.config.context_tokens) as span:
                context = assemble(reranked, self.config.context_tokens)
                span.attributes["tokens"] = context.tokens
                span.attributes["passages"] = len(context.passages)

        return RetrievalResult(
            question=question,
            standalone_question=standalone,
            candidates=tuple(reranked),
            context=context,
        )
