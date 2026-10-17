"""Offline, deterministic LLM stand-ins.

`ExtractiveLLM` is not a language model. It is a deterministic policy that
answers from the retrieved context by selecting and citing the sentences that
best cover the question, decides to abstain when coverage is too low, and
drives the agent loop's tool calls with a simple plan.

Its purpose is leverage, not quality: it makes the pipeline, the agent loop and
the eval harness runnable with no API key, no cost and no flakiness, so the
tests assert on *system* behaviour (citations resolve, abstention triggers,
budgets hold) rather than on model mood. Point `ARCHIVIST_LLM=openai` at the
same ports to get real answers.
"""

from __future__ import annotations

import json
import re

from archivist.retrieve.context import PASSAGE_SEPARATOR
from archivist.text import content_terms
from archivist.types import LLMMessage, LLMResponse, ToolCall

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_LINE_START = re.compile(r"^(\||[*\-+]\s|\d+[.)]\s|#{1,6}\s)")
_MARKER_BLOCK = re.compile(r"^\[(S\d+)\]\s*(.*?)\n(.*)$", re.DOTALL)


def _unwrap(text: str) -> str:
    """Join hard-wrapped prose lines, keeping list items and table rows intact.

    Source documents are often wrapped at 80 columns, which would otherwise
    split a sentence into fragments. A real model does not care; a sentence
    selector does.
    """
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        continues = bool(
            out
            and out[-1].strip()
            and stripped
            and not _LINE_START.match(stripped)
            and not _LINE_START.match(out[-1].strip())
        )
        if continues:
            out[-1] = f"{out[-1].rstrip()} {stripped}"
        else:
            out.append(stripped)
    return "\n".join(out)


def _parse_context(context: str) -> list[tuple[str, str, str]]:
    """Recover (marker, location, body) triples from an assembled context block.

    A real model reads the rendered prompt, so this stand-in does too - which
    means it must tolerate whatever preamble the prompt template adds around
    the passages.
    """
    passages: list[tuple[str, str, str]] = []
    for block in context.split(PASSAGE_SEPARATOR):
        start = block.find("[S")
        if start == -1:
            continue
        match = _MARKER_BLOCK.match(block[start:].strip())
        if match:
            passages.append((match.group(1), match.group(2).strip(), match.group(3).strip()))
    return passages


class ExtractiveLLM:
    """Selects the best-covering sentences from context and cites them."""

    def __init__(self, max_sentences: int = 3, min_coverage: float = 0.3) -> None:
        self.max_sentences = max_sentences
        self.min_coverage = min_coverage

    @property
    def name(self) -> str:
        return "extractive"

    def _answer(self, question: str, context: str) -> str:
        wanted = content_terms(question)
        if not wanted:
            return "INSUFFICIENT_EVIDENCE"

        scored: list[tuple[float, str, str]] = []
        for marker, location, body in _parse_context(context):
            # A table row or a terse line inherits the meaning of its heading,
            # so heading terms count toward coverage at half weight.
            heading = content_terms(location) & wanted
            for sentence in _SENTENCE.split(_unwrap(body)):
                cleaned = " ".join(sentence.split())
                if len(cleaned) < 20:
                    continue
                overlap = wanted & content_terms(cleaned)
                if not overlap:
                    continue
                inherited = heading - overlap
                coverage = (len(overlap) + 0.5 * len(inherited)) / len(wanted)
                # Prefer informative sentences, but penalise rambling ones.
                density = len(overlap) / max(8, len(cleaned.split()))
                scored.append((coverage + 0.5 * density, marker, cleaned))

        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored or scored[0][0] < self.min_coverage:
            return "INSUFFICIENT_EVIDENCE"

        lines: list[str] = []
        seen: set[str] = set()
        for _, marker, sentence in scored[: self.max_sentences]:
            if sentence in seen:
                continue
            seen.add(sentence)
            lines.append(f"{sentence} [{marker}]")
        return " ".join(lines)

    def _plan(self, messages: list[LLMMessage], tools: list[dict[str, object]]) -> ToolCall:
        """Search until there is context, then answer. Multi-hop via unmet terms."""
        tool_names = {str(tool.get("name")) for tool in tools}
        question = next(
            (message.content for message in messages if message.role == "user"), ""
        )
        observations = [message.content for message in messages if message.role == "tool"]

        if not observations and "search_corpus" in tool_names:
            return ToolCall(name="search_corpus", arguments={"query": question})

        covered: set[str] = set()
        for observation in observations:
            covered |= content_terms(observation)
        missing = content_terms(question) - covered

        searches = sum(1 for message in messages if message.role == "tool")
        if missing and searches < 2 and "search_corpus" in tool_names:
            return ToolCall(
                name="search_corpus",
                arguments={"query": " ".join(sorted(missing))},
            )
        return ToolCall(name="answer", arguments={"question": question})

    def complete(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, object]] | None = None,
    ) -> LLMResponse:
        prompt_tokens = sum(len(message.content.split()) for message in messages)
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": 0, "calls": 1}

        if tools:
            call = self._plan(messages, tools)
            return LLMResponse(text=json.dumps({"tool": call.name}), usage=usage, tool_call=call)

        question = next(
            (message.content for message in reversed(messages) if message.role == "user"), ""
        )
        context = PASSAGE_SEPARATOR.join(
            message.content for message in messages if message.role == "context"
        )

        text = self._answer(question, context)
        usage["completion_tokens"] = len(text.split())
        return LLMResponse(text=text, usage=usage)


class ScriptedLLM:
    """Replays fixed responses. For asserting on exact prompt/response flows."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[list[LLMMessage]] = []

    @property
    def name(self) -> str:
        return "scripted"

    def complete(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, object]] | None = None,  # noqa: ARG002 - part of the LLM port
    ) -> LLMResponse:
        self.prompts.append(list(messages))
        text = self._responses.pop(0) if self._responses else "INSUFFICIENT_EVIDENCE"
        return LLMResponse(text=text, usage={"calls": 1})
