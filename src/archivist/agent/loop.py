"""The agent loop.

Plain RAG is one retrieval and one generation. That is the right default and it
is what you should ship first. The loop here exists for the questions plain RAG
cannot answer: multi-hop ("which service owns the table that X writes to?"),
and queries whose useful search terms only appear after a first lookup.

The loop is deliberately bounded. Three budgets - steps, searches and an
explicit `answer` terminal tool - are what separate an agent from an expensive
infinite regress. Every step is traced, and the accumulated evidence is what
the final answer is generated from, so citations still resolve to real chunks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from archivist.agent.tools import ToolRegistry
from archivist.config import AgentConfig
from archivist.generate.answerer import Answerer
from archivist.obs.trace import Trace
from archivist.ports import LLM
from archivist.retrieve.context import assemble
from archivist.retrieve.pipeline import RetrievalPipeline
from archivist.retrieve.query import rewrite_question
from archivist.types import Answer, LLMMessage, Scored

AGENT_SYSTEM_PROMPT = """You research a question against a private corpus.

Use search_corpus to find evidence. Search again with different terms only if
the question has a part you have no evidence for yet. Call answer as soon as
the evidence is sufficient. Never answer from prior knowledge.
"""

ANSWER_TOOL: dict[str, object] = {
    "name": "answer",
    "description": "Finish and write the grounded answer from the evidence gathered.",
    "parameters": {
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"],
    },
}


@dataclass
class AgentStep:
    index: int
    tool: str
    arguments: dict[str, object]
    observation: str


@dataclass
class AgentRun:
    answer: Answer
    steps: list[AgentStep] = field(default_factory=list)
    trace: Trace = field(default_factory=Trace)

    @property
    def search_count(self) -> int:
        return sum(1 for step in self.steps if step.tool == "search_corpus")


class ResearchAgent:
    def __init__(
        self,
        llm: LLM,
        pipeline: RetrievalPipeline,
        tools: ToolRegistry,
        answerer: Answerer,
        config: AgentConfig | None = None,
    ) -> None:
        self.llm = llm
        self.pipeline = pipeline
        self.tools = tools
        self.answerer = answerer
        self.config = config or AgentConfig()

    def _tool_specs(self, searches_used: int) -> list[dict[str, object]]:
        specs = [
            spec
            for spec in self.tools.specs()
            if spec["name"] != "search_corpus" or searches_used < self.config.max_searches
        ]
        return [*specs, ANSWER_TOOL]

    def _finalise(
        self,
        question: str,
        evidence: list[Scored],
        trace: Trace,
    ) -> Answer:
        budget = self.pipeline.config.context_tokens
        deduped: dict[str, Scored] = {}
        for scored in sorted(evidence, key=lambda item: item.score, reverse=True):
            deduped.setdefault(scored.chunk.chunk_id, scored)
        context = assemble(list(deduped.values()), budget)
        return self.answerer.answer(question, context, trace=trace)

    def run(self, question: str, history: list[str] | None = None) -> AgentRun:
        trace = Trace()
        steps: list[AgentStep] = []
        evidence: list[Scored] = []
        with trace.span("agent.rewrite"):
            question = rewrite_question(question, history or [], self.llm)
        messages = [
            LLMMessage(role="system", content=AGENT_SYSTEM_PROMPT),
            LLMMessage(role="user", content=question),
        ]

        with trace.span("agent", question=question[:80], max_steps=self.config.max_steps):
            for index in range(1, self.config.max_steps + 1):
                searches_used = sum(1 for step in steps if step.tool == "search_corpus")
                available = self._tool_specs(searches_used)
                with trace.span("agent.step", index=index) as span:
                    response = self.llm.complete(messages, tools=available)
                    trace.record_usage(response.usage)
                    call = response.tool_call

                    if call is None or call.name == "answer":
                        span.attributes["tool"] = "answer"
                        answer = self._finalise(question, evidence, trace)
                        return AgentRun(answer=answer, steps=steps, trace=trace)

                    span.attributes["tool"] = call.name
                    if call.name not in {str(spec["name"]) for spec in available}:
                        # Budgets are enforced here, not by trusting the model to
                        # stop asking: a withdrawn tool is refused and the refusal
                        # is fed back as an observation.
                        refusal = (
                            f"{call.name} is no longer available "
                            f"(search budget of {self.config.max_searches} used). "
                            "Call answer with the evidence you have."
                        )
                        span.attributes["refused"] = True
                        steps.append(
                            AgentStep(
                                index=index,
                                tool=f"{call.name}:refused",
                                arguments=dict(call.arguments),
                                observation=refusal,
                            )
                        )
                        messages.append(LLMMessage(role="tool", content=refusal))
                        continue

                    result = self.tools.call(call.name, call.arguments)
                    evidence.extend(result.chunks)
                    span.attributes["evidence"] = len(evidence)
                    steps.append(
                        AgentStep(
                            index=index,
                            tool=call.name,
                            arguments=dict(call.arguments),
                            observation=result.content[:2000],
                        )
                    )
                    messages.append(
                        LLMMessage(role="tool", content=f"{call.name} -> {result.content}")
                    )

            # Budget exhausted: answer from whatever evidence exists rather than
            # looping forever or returning nothing.
            with trace.span("agent.budget_exhausted", steps=len(steps)):
                answer = self._finalise(question, evidence, trace)
        return AgentRun(answer=answer, steps=steps, trace=trace)


def plain_rag_answer(
    question: str,
    pipeline: RetrievalPipeline,
    answerer: Answerer,
    history: list[str] | None = None,
) -> Answer:
    """The non-agentic baseline: retrieve once, answer once. The thing to beat."""
    trace = Trace()
    result = pipeline.run(question, history=history, trace=trace)
    return answerer.answer(result.standalone_question, result.context, trace=trace)
