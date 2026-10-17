"""Query understanding.

Two cheap transforms carry most of the value:

* Rewriting a follow-up question into a standalone one. "And what about the
  retention period?" retrieves nothing on its own; resolved against history it
  retrieves the right chunk.
* Generating a small number of paraphrases, retrieving for each, and fusing the
  results. This buys recall on vocabulary mismatch without a bigger `k`.

Both are LLM-shaped operations with a deterministic fallback, so the pipeline
behaves sensibly offline.
"""

from __future__ import annotations

import re

from archivist.ports import LLM
from archivist.text import tokenize
from archivist.types import LLMMessage

_FOLLOWUP_MARKERS = re.compile(r"^(and|what about|how about|why|also|then)\b", re.IGNORECASE)
_PRONOUNS = frozenset({"it", "its", "they", "them", "their", "that", "this", "those", "there"})

REWRITE_PROMPT = """Rewrite the user's latest question so it can be understood \
without the conversation. Keep it short, keep every proper noun, and do not answer it.

Conversation:
{history}

Latest question: {question}

Standalone question:"""


def needs_rewrite(question: str, history: list[str]) -> bool:
    if not history:
        return False
    if _FOLLOWUP_MARKERS.match(question.strip()):
        return True
    return any(token in _PRONOUNS for token in tokenize(question))


def rewrite_question(question: str, history: list[str], llm: LLM | None = None) -> str:
    """Resolve a follow-up into a standalone question."""
    if not needs_rewrite(question, history):
        return question
    if llm is None:
        # Deterministic fallback: prepend the most recent user turn for context.
        return f"{history[-1].strip()} {question.strip()}".strip()
    prompt = REWRITE_PROMPT.format(history="\n".join(history[-4:]), question=question)
    response = llm.complete([LLMMessage(role="user", content=prompt)])
    rewritten = response.text.strip().splitlines()[0].strip()
    return rewritten or question


def expand_question(question: str, count: int = 1, llm: LLM | None = None) -> list[str]:
    """Return the question plus up to `count` additional probes of the same intent."""
    variants = [question]
    if count <= 0:
        return variants
    if llm is not None:
        prompt = (
            f"Write {count} alternative phrasings of this question, one per line, "
            f"using different vocabulary but the same meaning:\n{question}"
        )
        response = llm.complete([LLMMessage(role="user", content=prompt)])
        for line in response.text.splitlines():
            candidate = line.strip(" -*0123456789.").strip()
            if candidate and candidate.lower() != question.lower():
                variants.append(candidate)
        return variants[: count + 1]

    # Fallback: a keyword-only probe. Strips the question framing that dense
    # retrievers waste capacity on and lexical search scores as noise.
    keywords = [token for token in tokenize(question) if len(token) > 3]
    if keywords:
        variants.append(" ".join(dict.fromkeys(keywords)))
    return variants[: count + 1]
