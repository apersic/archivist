from __future__ import annotations

from archivist.app import Archivist
from archivist.generate.answerer import Answerer, extract_markers
from archivist.llm.extractive import ExtractiveLLM, ScriptedLLM
from archivist.obs.trace import Trace
from archivist.retrieve.context import assemble
from archivist.types import AnswerStatus, Chunk, Scored


def _context(*texts: str):
    candidates = [
        Scored(
            chunk=Chunk(
                chunk_id=f"c{index}",
                doc_id="doc-1",
                text=text,
                ordinal=index,
                source="guide.md",
                title="Guide",
                section="Limits",
                token_estimate=len(text.split()),
            ),
            score=1.0 - index * 0.1,
        )
        for index, text in enumerate(texts)
    ]
    return assemble(candidates, budget_tokens=1000)


def test_markers_are_extracted_once_each() -> None:
    assert extract_markers("a [S1] b [S2] c [S1]") == ["S1", "S2"]


def test_answer_cites_the_passage_it_used() -> None:
    context = _context("The per-key limit is 2000 events per second.")
    answer = Answerer(ExtractiveLLM()).answer("What is the per-key limit?", context)

    assert answer.status is AnswerStatus.ANSWERED
    assert "2000" in answer.text
    assert answer.cited_chunk_ids == ("c0",)


def test_empty_context_abstains_without_calling_the_model() -> None:
    llm = ScriptedLLM(["this should never be used"])
    answer = Answerer(llm).answer("anything?", assemble([], budget_tokens=100))

    assert answer.status is AnswerStatus.ABSTAINED
    assert llm.prompts == []


def test_model_abstention_is_propagated() -> None:
    context = _context("Unrelated text about backups and encryption.")
    answer = Answerer(ScriptedLLM(["INSUFFICIENT_EVIDENCE"])).answer("What is the limit?", context)
    assert answer.status is AnswerStatus.ABSTAINED


def test_invented_citations_are_stripped() -> None:
    context = _context("The per-key limit is 2000 events per second.")
    llm = ScriptedLLM(["The limit is 2000 [S1]. Also something invented [S9]."])
    answer = Answerer(llm).answer("What is the limit?", context)

    assert answer.status is AnswerStatus.ANSWERED
    assert "[S9]" not in answer.text
    assert answer.cited_chunk_ids == ("c0",)


def test_uncited_answers_are_downgraded_to_abstention() -> None:
    context = _context("The per-key limit is 2000 events per second.")
    answer = Answerer(ScriptedLLM(["The limit is 2000 events per second."])).answer(
        "What is the limit?", context
    )
    assert answer.status is AnswerStatus.ABSTAINED


def test_answers_are_grounded_in_retrieved_context(app: Archivist) -> None:
    answer = app.ask("What is the retention period for events_raw?")
    assert answer.status is AnswerStatus.ANSWERED
    assert "30 days" in answer.text
    assert all(citation.chunk_id in {s.chunk.chunk_id for s in answer.contexts}
               for citation in answer.citations)


def test_trace_records_every_stage(app: Archivist) -> None:
    trace = Trace()
    result = app.pipeline.run("What is the p95 latency budget?", trace=trace)
    app.answerer.answer(result.standalone_question, result.context, trace=trace)

    names = [span.name for span in trace.spans]
    for expected in ("retrieve.rewrite", "retrieve.dense", "retrieve.lexical", "retrieve.fuse",
                     "retrieve.rerank", "retrieve.assemble", "generate"):
        assert expected in names
    assert all(span.duration_ms is not None for span in trace.spans)
