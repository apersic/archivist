"""The eval harness.

This is the component that turns the project from a demo into engineering. It
runs the golden set through a named strategy, reports retrieval and generation
metrics side by side, and compares against a stored baseline so a change that
improves one metric and quietly wrecks another cannot land unnoticed.

Baselines live in `evals/baselines/<strategy>.json` and are committed, so the
diff of a tuning PR shows both the config change and its measured effect.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean

from archivist.app import Archivist
from archivist.evaluation.dataset import EvalCase
from archivist.evaluation.metrics import (
    answer_match,
    citation_support,
    groundedness,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from archivist.types import Answer, AnswerStatus

Strategy = Callable[[str, list[str]], Answer]


@dataclass(frozen=True, slots=True)
class CaseResult:
    question: str
    status: str
    source_recall: float
    mrr: float
    ndcg: float
    groundedness: float
    citation_support: float
    answer_match: float
    abstention_correct: bool
    tags: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        if not self.abstention_correct:
            return False
        if self.status == AnswerStatus.ABSTAINED.value:
            return True
        return self.answer_match >= 0.5 and self.groundedness >= 0.6


@dataclass
class EvalReport:
    strategy: str
    k: int
    cases: list[CaseResult] = field(default_factory=list)
    config_fingerprint: str = ""

    def aggregate(self) -> dict[str, float]:
        if not self.cases:
            return {}
        answered = [case for case in self.cases if case.status == AnswerStatus.ANSWERED.value]
        return {
            f"source_recall@{self.k}": mean(case.source_recall for case in self.cases),
            "mrr": mean(case.mrr for case in self.cases),
            f"ndcg@{self.k}": mean(case.ndcg for case in self.cases),
            "groundedness": mean(case.groundedness for case in self.cases),
            "citation_support": (
                mean(case.citation_support for case in answered) if answered else 1.0
            ),
            "answer_match": mean(case.answer_match for case in self.cases),
            "abstention_accuracy": mean(
                1.0 if case.abstention_correct else 0.0 for case in self.cases
            ),
            "pass_rate": mean(1.0 if case.passed else 0.0 for case in self.cases),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "k": self.k,
            "config_fingerprint": self.config_fingerprint,
            "metrics": self.aggregate(),
            "cases": [asdict(case) for case in self.cases],
        }

    def render(self) -> str:
        lines = [f"strategy: {self.strategy}  cases: {len(self.cases)}"]
        for metric, value in self.aggregate().items():
            lines.append(f"  {metric:<22} {value:.3f}")
        failures = [case for case in self.cases if not case.passed]
        if failures:
            lines.append(f"  failing cases ({len(failures)}):")
            lines.extend(
                f"    - [{case.status}] {case.question}"
                f" (match={case.answer_match:.2f} recall={case.source_recall:.2f})"
                for case in failures
            )
        return "\n".join(lines)


def evaluate(
    app: Archivist,
    cases: list[EvalCase],
    strategy: Strategy | None = None,
    strategy_name: str = "plain-rag",
    k: int = 5,
) -> EvalReport:
    run = strategy or (lambda question, history: app.ask(question, history=history))
    report = EvalReport(
        strategy=strategy_name, k=k, config_fingerprint=app.settings.fingerprint()
    )

    for case in cases:
        answer = run(case.question, list(case.history))
        # Labels are at document granularity, so retrieval is scored there too:
        # scoring chunk ids against a whole document's chunks would cap recall
        # at k/chunks_per_document and make the number meaningless.
        retrieved = list(dict.fromkeys(scored.chunk.source for scored in answer.contexts))
        relevant = set(case.relevant_sources)
        abstained = answer.status is AnswerStatus.ABSTAINED
        report.cases.append(
            CaseResult(
                question=case.question,
                status=answer.status.value,
                source_recall=recall_at_k(retrieved, relevant, k) if relevant else 1.0,
                mrr=mean_reciprocal_rank(retrieved, relevant) if relevant else 1.0,
                ndcg=ndcg_at_k(retrieved, relevant, k) if relevant else 1.0,
                groundedness=groundedness(answer),
                citation_support=citation_support(answer),
                answer_match=(
                    1.0
                    if case.unanswerable
                    else answer_match(answer, list(case.must_contain))
                ),
                abstention_correct=abstained == case.unanswerable,
                tags=case.tags,
            )
        )
    return report


def compare(report: EvalReport, baseline_path: Path, tolerance: float = 0.02) -> list[str]:
    """Return human-readable regressions against a stored baseline."""
    if not baseline_path.exists():
        return []
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")).get("metrics", {})
    current = report.aggregate()
    regressions: list[str] = []
    for metric, previous in baseline.items():
        now = current.get(metric)
        if now is None:
            continue
        if now < previous - tolerance:
            regressions.append(f"{metric}: {previous:.3f} -> {now:.3f}")
    return regressions


def save_report(report: EvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
