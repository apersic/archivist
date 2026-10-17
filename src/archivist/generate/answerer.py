"""Grounded answer generation.

The generator is the last place where a hallucination can be caught cheaply, so
this module does three things beyond calling the model:

* refuses to call it at all when retrieval returned nothing;
* parses the markers the model emitted and keeps only those that exist in the
  context, which turns a fabricated citation into a dropped one;
* downgrades an uncited answer to an abstention when `require_citations` is on,
  because an answer nobody can check is worse than no answer.
"""

from __future__ import annotations

import re

from archivist.generate.prompts import (
    ABSTENTION_MESSAGE,
    ABSTENTION_TEXT,
    ANSWER_CONTEXT_TEMPLATE,
    ANSWER_SYSTEM_PROMPT,
)
from archivist.obs.trace import Trace
from archivist.ports import LLM
from archivist.retrieve.context import AssembledContext
from archivist.types import Answer, AnswerStatus, Citation, LLMMessage

_MARKER = re.compile(r"\[(S\d+)\]")


def extract_markers(text: str) -> list[str]:
    return list(dict.fromkeys(_MARKER.findall(text)))


class Answerer:
    def __init__(self, llm: LLM, require_citations: bool = True) -> None:
        self.llm = llm
        self.require_citations = require_citations

    def _abstain(self, question: str, context: AssembledContext, usage: dict[str, int]) -> Answer:
        return Answer(
            question=question,
            text=ABSTENTION_MESSAGE,
            status=AnswerStatus.ABSTAINED,
            contexts=context.passages,
            usage=usage,
        )

    def answer(
        self,
        question: str,
        context: AssembledContext,
        trace: Trace | None = None,
    ) -> Answer:
        trace = trace or Trace()
        with trace.span("generate", llm=self.llm.name, passages=len(context.passages)) as span:
            if context.is_empty:
                span.attributes["outcome"] = "no-context"
                return self._abstain(question, context, {})

            messages = [
                LLMMessage(role="system", content=ANSWER_SYSTEM_PROMPT),
                LLMMessage(
                    role="context",
                    content=ANSWER_CONTEXT_TEMPLATE.format(context=context.text),
                ),
                LLMMessage(role="user", content=question),
            ]
            response = self.llm.complete(messages)
            trace.record_usage(response.usage)
            text = response.text.strip()

            if ABSTENTION_TEXT in text:
                span.attributes["outcome"] = "abstained"
                return self._abstain(question, context, response.usage)

            citations: list[Citation] = []
            for marker in extract_markers(text):
                citation = context.citation_for(marker)
                if citation is None:
                    # The model invented a marker: drop it rather than show it.
                    text = text.replace(f"[{marker}]", "").replace("  ", " ")
                    continue
                citations.append(citation)

            if self.require_citations and not citations:
                span.attributes["outcome"] = "uncited"
                return self._abstain(question, context, response.usage)

            span.attributes["outcome"] = "answered"
            span.attributes["citations"] = len(citations)
            return Answer(
                question=question,
                text=text.strip(),
                status=AnswerStatus.ANSWERED,
                citations=tuple(citations),
                contexts=context.passages,
                usage=response.usage,
            )
