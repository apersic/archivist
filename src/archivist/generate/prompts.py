"""Prompts, kept in one place so they can be versioned and diffed.

The system prompt encodes three rules that matter more than wording: answer
only from the context, cite every claim with a marker, and say
INSUFFICIENT_EVIDENCE rather than guess. The last one is what makes the system
trustworthy; the eval harness scores it explicitly.
"""

from __future__ import annotations

ANSWER_SYSTEM_PROMPT = """You answer questions strictly from the provided sources.

Rules:
1. Use only the sources. Do not use prior knowledge, and do not speculate.
2. Cite every factual sentence with the marker of its source, like [S2].
   Cite more than one marker when a sentence draws on several sources.
3. If the sources do not contain the answer, reply with exactly:
   INSUFFICIENT_EVIDENCE
4. Be concise: at most four sentences. Quote exact names, numbers and
   identifiers rather than paraphrasing them.
"""

ANSWER_CONTEXT_TEMPLATE = """Sources:

{context}"""

ABSTENTION_TEXT = "INSUFFICIENT_EVIDENCE"

ABSTENTION_MESSAGE = (
    "I could not find this in the indexed corpus. Try rephrasing, or add the "
    "source document and re-index."
)
