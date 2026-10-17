from __future__ import annotations

from archivist.agent.loop import ResearchAgent
from archivist.app import Archivist
from archivist.config import AgentConfig
from archivist.ports import LLM
from archivist.types import AnswerStatus, LLMMessage, LLMResponse, ToolCall


class LoopingLLM:
    """A model that only ever wants to search. Exercises the step budget."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "looping"

    def complete(
        self,
        messages: list[LLMMessage],  # noqa: ARG002 - part of the LLM port
        tools: list[dict[str, object]] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        if not tools:
            return LLMResponse(text="INSUFFICIENT_EVIDENCE", usage={"calls": 1})
        return LLMResponse(
            text="",
            usage={"calls": 1},
            tool_call=ToolCall(name="search_corpus", arguments={"query": "anything"}),
        )


def _agent(app: Archivist, llm: LLM, config: AgentConfig) -> ResearchAgent:
    return ResearchAgent(
        llm=llm,
        pipeline=app.pipeline,
        tools=app.tools,
        answerer=app.answerer,
        config=config,
    )


def test_tool_registry_exposes_specs_for_the_model(app: Archivist) -> None:
    names = {spec["name"] for spec in app.tools.specs()}
    assert names == {"search_corpus", "fetch_chunk", "list_sources"}


def test_unknown_tool_is_an_observation_not_a_crash(app: Archivist) -> None:
    result = app.tools.call("teleport", {"to": "mars"})
    assert "unknown tool" in result.content


def test_missing_arguments_are_reported_back_to_the_model(app: Archivist) -> None:
    result = app.tools.call("search_corpus", {})
    assert "missing arguments: query" in result.content


def test_search_tool_returns_citable_chunks(app: Archivist) -> None:
    result = app.tools.call("search_corpus", {"query": "retention of events_raw", "limit": 3})
    assert result.chunks
    assert len(result.chunks) <= 3
    assert all(app.vector_index.get(scored.chunk.chunk_id) for scored in result.chunks)


def test_fetch_chunk_rejects_unknown_ids(app: Archivist) -> None:
    assert "no such chunk" in app.tools.call("fetch_chunk", {"chunk_id": "nope"}).content


def test_agent_searches_then_answers_with_citations(app: Archivist) -> None:
    run = app.agent.run("What is the p95 latency budget for Kestrel?")

    assert run.search_count >= 1
    assert run.answer.status is AnswerStatus.ANSWERED
    assert run.answer.citations
    assert "180 ms" in run.answer.text


def test_step_budget_is_enforced(app: Archivist) -> None:
    llm = LoopingLLM()
    run = _agent(app, llm, AgentConfig(max_steps=3, max_searches=10)).run("loop forever")

    assert len(run.steps) == 3
    assert run.answer.status in {AnswerStatus.ANSWERED, AnswerStatus.ABSTAINED}


def test_search_budget_is_enforced_even_if_the_model_ignores_it(app: Archivist) -> None:
    llm = LoopingLLM()
    agent = _agent(app, llm, AgentConfig(max_steps=5, max_searches=2))
    run = agent.run("loop forever")

    assert run.search_count == 2, "the loop must refuse searches beyond the budget"
    assert [step.tool for step in run.steps[2:]] == ["search_corpus:refused"] * 3
    assert "no longer available" in run.steps[-1].observation


def test_agent_evidence_is_deduplicated_before_answering(app: Archivist) -> None:
    run = app.agent.run("Which job writes events_curated and who owns it?")
    chunk_ids = [scored.chunk.chunk_id for scored in run.answer.contexts]
    assert len(chunk_ids) == len(set(chunk_ids))
