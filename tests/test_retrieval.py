from __future__ import annotations

import pytest

from archivist.app import Archivist, build_indexes
from archivist.config import RetrievalConfig
from archivist.index.embeddings import HashingEmbedder
from archivist.index.hybrid import reciprocal_rank_fusion
from archivist.index.lexical import BM25Index
from archivist.retrieve.context import assemble
from archivist.retrieve.pipeline import RetrievalPipeline
from archivist.retrieve.query import needs_rewrite, rewrite_question
from archivist.retrieve.rerank import IdentityReranker, LexicalOverlapReranker
from archivist.types import Chunk, Scored


def _chunk(chunk_id: str, text: str, ordinal: int = 0, source: str = "a.md") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        text=text,
        ordinal=ordinal,
        source=source,
        token_estimate=len(text.split()),
    )


def test_embeddings_are_normalised_and_deterministic() -> None:
    embedder = HashingEmbedder(dimensions=128)
    first = embedder.embed_query("rate limit errors")
    second = embedder.embed_query("rate limit errors")
    assert first == second
    assert pytest.approx(sum(value * value for value in first), abs=1e-9) == 1.0


def test_lexical_index_finds_exact_identifiers() -> None:
    index = BM25Index()
    index.add(
        [
            _chunk("c1", "The service returns PEL-429 when the caller is rate limited."),
            _chunk("c2", "Latency budgets are documented per endpoint."),
        ]
    )
    hits = index.search("PEL-429", limit=2)
    assert hits[0].chunk.chunk_id == "c1"


def test_fusion_rewards_agreement_between_retrievers() -> None:
    dense = [Scored(_chunk("c1", "x"), 0.9, "dense"), Scored(_chunk("c2", "y"), 0.8, "dense")]
    lexical = [Scored(_chunk("c2", "y"), 7.0, "lexical"), Scored(_chunk("c3", "z"), 3.0, "lexical")]

    fused = reciprocal_rank_fusion([dense, lexical], k=10)

    assert fused[0].chunk.chunk_id == "c2", "agreed-upon chunk should win"
    assert fused[0].retriever == "dense+lexical"
    assert {scored.chunk.chunk_id for scored in fused} == {"c1", "c2", "c3"}


def test_reranker_promotes_full_query_coverage() -> None:
    candidates = [
        # Ranked first by fusion, but it covers only one of the two query terms.
        Scored(_chunk("c1", "retention applies to every table and to all backups"), 0.5, "dense"),
        Scored(_chunk("c2", "events_raw retention is 30 days, enforced nightly"), 0.4, "dense"),
    ]
    reranked = LexicalOverlapReranker().rerank("events_raw retention", candidates, limit=2)
    assert reranked[0].chunk.chunk_id == "c2"


def test_context_assembly_respects_the_token_budget() -> None:
    candidates = [
        Scored(_chunk(f"c{index}", " ".join(["word"] * 100), ordinal=index), 1.0 / (index + 1))
        for index in range(10)
    ]
    context = assemble(candidates, budget_tokens=250)

    assert context.tokens <= 300
    assert len(context.passages) < len(candidates)
    assert [citation.marker for citation in context.citations] == [
        f"S{index + 1}" for index in range(len(context.passages))
    ]


def test_context_assembly_drops_near_duplicates() -> None:
    text = "events_raw is retained for exactly thirty days and then dropped by partition"
    candidates = [
        Scored(_chunk("c1", text), 1.0),
        Scored(_chunk("c2", text + " nightly"), 0.9),
        Scored(_chunk("c3", "kestrel p95 latency budget is 180 ms"), 0.8),
    ]
    context = assemble(candidates, budget_tokens=1000)
    assert [scored.chunk.chunk_id for scored in context.passages] == ["c1", "c3"]


def test_followup_questions_are_rewritten_against_history() -> None:
    assert needs_rewrite("And what about its retention?", ["How is events_raw written?"])
    assert not needs_rewrite("What is the retention of events_raw?", [])
    rewritten = rewrite_question("And what about its retention?", ["How is events_raw written?"])
    assert "events_raw" in rewritten


def test_pipeline_finds_the_right_document_for_a_specific_question(app: Archivist) -> None:
    result = app.pipeline.run("What is the p95 latency budget for Kestrel search?")
    sources = [scored.chunk.source for scored in result.candidates]
    assert sources[0] == "platform/kestrel-search.md"
    assert "180 ms" in result.context.text


def test_hybrid_is_never_worse_than_dense_only_on_identifier_lookup(app: Archivist) -> None:
    """The lexical half earns its keep on exact-string queries.

    The gap is modest here because the default `HashingEmbedder` indexes
    character n-grams, so it already matches identifiers; with a semantic
    embedder the dense half degrades much further and this test gets louder.
    """
    config = RetrievalConfig(expand_neighbours=False, query_expansions=0, rerank_limit=3)

    def identifier_hits(lexical_index: BM25Index) -> int:
        pipeline = RetrievalPipeline(
            embedder=app.embedder,
            vector_index=app.vector_index,
            lexical_index=lexical_index,
            reranker=IdentityReranker(),
            config=config,
        )
        result = pipeline.run("PEL-430")
        return sum(1 for scored in result.candidates if "PEL-430" in scored.chunk.text)

    dense_only = identifier_hits(BM25Index())
    hybrid = identifier_hits(app.lexical_index)
    assert hybrid == 3
    assert hybrid >= dense_only


def test_indexes_are_built_from_the_same_chunks() -> None:
    chunks = [_chunk("c1", "alpha beta"), _chunk("c2", "gamma delta", ordinal=1)]
    vector_index, lexical_index = build_indexes(chunks, HashingEmbedder(dimensions=64))
    assert len(vector_index) == len(lexical_index) == 2
    assert vector_index.get("c1") is not None


def test_neighbour_expansion_returns_adjacent_chunks() -> None:
    chunks = [_chunk(f"c{index}", f"text {index}", ordinal=index) for index in range(4)]
    vector_index, _ = build_indexes(chunks, HashingEmbedder(dimensions=64))
    neighbour_ids = {chunk.chunk_id for chunk in vector_index.neighbours("c2", window=1)}
    assert neighbour_ids == {"c1", "c3"}
