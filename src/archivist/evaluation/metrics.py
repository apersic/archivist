"""Metrics.

Retrieval and generation fail differently and must be measured separately. If
you only score the final answer you cannot tell a retrieval miss from a
generation miss, and you will tune prompts to fix an indexing bug.

Retrieval: recall@k, MRR, nDCG@k over the chunk ids a case declares relevant.
Generation: groundedness (is every sentence supported by the context?),
citation precision (do the cited passages actually support the answer?), and
abstention correctness (did it refuse exactly when it should have?).

Groundedness here is a lexical-overlap proxy so it runs offline and for free.
`LLMJudge` implements the same signature for when you want a model's opinion;
calibrate the proxy against it on a sample before trusting either.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2

from archivist.ports import LLM
from archivist.text import content_terms
from archivist.types import Answer, AnswerStatus, LLMMessage


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0
    top = set(retrieved[:k])
    return len(top & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for chunk_id in top if chunk_id in relevant) / len(top)


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0
    gains = sum(
        1.0 / log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved[:k], start=1)
        if chunk_id in relevant
    )
    ideal = sum(1.0 / log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return gains / ideal if ideal else 0.0


def groundedness(answer: Answer) -> float:
    """Fraction of the answer's content terms that appear in the retrieved context.

    A cheap, deterministic hallucination detector: a sentence built from terms
    that are nowhere in the context did not come from the corpus.
    """
    if answer.status is not AnswerStatus.ANSWERED:
        return 1.0
    claim_terms = content_terms(answer.text)
    if not claim_terms:
        return 1.0
    context_terms: set[str] = set()
    for scored in answer.contexts:
        context_terms |= content_terms(scored.chunk.text)
    return len(claim_terms & context_terms) / len(claim_terms)


def citation_support(answer: Answer) -> float:
    """Fraction of cited passages that share meaningful content with the answer."""
    if not answer.citations:
        return 0.0 if answer.status is AnswerStatus.ANSWERED else 1.0
    by_id = {scored.chunk.chunk_id: scored.chunk for scored in answer.contexts}
    claim_terms = content_terms(answer.text)
    if not claim_terms:
        return 0.0
    supported = 0
    for citation in answer.citations:
        chunk = by_id.get(citation.chunk_id)
        if chunk is None:
            continue
        if content_terms(chunk.text) & claim_terms:
            supported += 1
    return supported / len(answer.citations)


def answer_match(answer: Answer, expected_terms: list[str]) -> float:
    """Fraction of the facts a correct answer must contain that are present."""
    if not expected_terms:
        return 1.0
    text = answer.text.lower()
    return sum(1 for term in expected_terms if term.lower() in text) / len(expected_terms)


@dataclass(frozen=True, slots=True)
class LLMJudge:
    """Model-graded faithfulness, behind the same call signature as the proxy."""

    llm: LLM
    prompt: str = (
        "You are grading an answer for faithfulness to its sources.\n"
        "Reply with a single number between 0 and 1 and nothing else.\n\n"
        "Sources:\n{context}\n\nAnswer:\n{answer}\n\nScore:"
    )

    def __call__(self, answer: Answer) -> float:
        context = "\n\n".join(scored.chunk.text for scored in answer.contexts)
        response = self.llm.complete(
            [
                LLMMessage(
                    role="user",
                    content=self.prompt.format(context=context, answer=answer.text),
                )
            ]
        )
        try:
            return max(0.0, min(1.0, float(response.text.strip().split()[0])))
        except (ValueError, IndexError):
            return 0.0
