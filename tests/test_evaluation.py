from __future__ import annotations

import json
from pathlib import Path

import pytest

from archivist.app import Archivist
from archivist.evaluation.dataset import EvalCase, load_cases, propose_cases, save_cases
from archivist.evaluation.harness import compare, evaluate, save_report
from archivist.evaluation.metrics import (
    answer_match,
    citation_support,
    groundedness,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from archivist.types import Answer, AnswerStatus, Chunk, Citation, Scored


def _answer(text: str, context_text: str, cited: bool = True) -> Answer:
    chunk = Chunk(
        chunk_id="c1", doc_id="d1", text=context_text, ordinal=0, source="a.md", title="A"
    )
    return Answer(
        question="q",
        text=text,
        status=AnswerStatus.ANSWERED,
        citations=(Citation(marker="S1", chunk_id="c1", location="A"),) if cited else (),
        contexts=(Scored(chunk=chunk, score=1.0),),
    )


def test_ranking_metrics_agree_on_a_known_ranking() -> None:
    retrieved = ["a", "b", "c", "d"]
    relevant = {"c"}
    assert recall_at_k(retrieved, relevant, 2) == 0.0
    assert recall_at_k(retrieved, relevant, 3) == 1.0
    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(1 / 3)
    assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)
    assert ndcg_at_k(retrieved, relevant, 4) == pytest.approx(0.5)


def test_perfect_ranking_scores_one() -> None:
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)
    assert mean_reciprocal_rank(["a"], {"a"}) == 1.0


def test_groundedness_catches_terms_absent_from_context() -> None:
    supported = _answer("The limit is 2000 events.", "The limit is 2000 events per second.")
    fabricated = _answer("The limit is set by Jupiter's moons.", "The limit is 2000 events.")
    assert groundedness(supported) == pytest.approx(1.0)
    assert groundedness(fabricated) < 0.6


def test_citation_support_detects_an_unrelated_citation() -> None:
    aligned = _answer("The limit is 2000 events.", "The limit is 2000 events per second.")
    misaligned = _answer("Backups are encrypted nightly.", "Kestrel caches results for 60 seconds.")
    assert citation_support(aligned) == 1.0
    assert citation_support(misaligned) == 0.0


def test_answer_match_is_case_insensitive() -> None:
    answer = _answer("Returns PEL-429 on rate limit.", "irrelevant")
    assert answer_match(answer, ["pel-429"]) == 1.0
    assert answer_match(answer, ["pel-429", "PEL-430"]) == 0.5


def test_golden_set_loads_and_round_trips(tmp_path: Path, project_root: Path) -> None:
    cases = load_cases(project_root / "evals" / "qa.jsonl")
    assert len(cases) >= 10
    assert any(case.unanswerable for case in cases), "a golden set needs unanswerable cases"

    out = tmp_path / "copy.jsonl"
    save_cases(cases, out)
    assert load_cases(out) == cases


def test_malformed_golden_line_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"question": "ok"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        load_cases(path)


def test_proposed_cases_are_labelled_as_drafts(app: Archivist) -> None:
    drafts = propose_cases(app.vector_index.all_chunks(), per_source=1)
    assert len(drafts) == len({chunk.source for chunk in app.vector_index.all_chunks()})
    assert all("draft" in case.tags for case in drafts)


def test_evaluate_reports_retrieval_and_generation_separately(app: Archivist) -> None:
    cases = [
        EvalCase(
            question="What is the p95 latency budget for Kestrel search?",
            relevant_sources=("platform/kestrel-search.md",),
            must_contain=("180",),
        ),
        EvalCase(question="What is the CEO's home address?", unanswerable=True),
    ]
    report = evaluate(app, cases, k=5)
    metrics = report.aggregate()

    assert metrics["source_recall@5"] == 1.0
    assert metrics["answer_match"] >= 0.5
    assert report.cases[0].status == AnswerStatus.ANSWERED.value
    assert report.config_fingerprint


def test_regression_comparison_flags_a_drop(tmp_path: Path, app: Archivist) -> None:
    cases = [
        EvalCase(
            question="What is the p95 latency budget for Kestrel search?",
            relevant_sources=("platform/kestrel-search.md",),
            must_contain=("180",),
        )
    ]
    report = evaluate(app, cases, k=5)
    baseline = tmp_path / "baseline.json"
    save_report(report, baseline)

    assert compare(report, baseline) == []

    inflated = json.loads(baseline.read_text(encoding="utf-8"))
    inflated["metrics"]["answer_match"] = 1.0
    inflated["metrics"]["source_recall@5"] = 1.0
    baseline.write_text(json.dumps(inflated), encoding="utf-8")

    degraded = evaluate(app, [EvalCase(question="unrelated nonsense", must_contain=("zzz",))], k=5)
    regressions = compare(degraded, baseline)
    assert any("answer_match" in regression for regression in regressions)
