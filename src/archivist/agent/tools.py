"""Tool registry.

A tool is a name, a JSON schema and a callable. Keeping the schema next to the
implementation means the prompt the model sees and the code that runs can never
drift apart, and the loop can validate arguments before executing anything.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from archivist.index.vector_store import InMemoryVectorIndex
from archivist.retrieve.pipeline import RetrievalPipeline
from archivist.types import Scored, ToolResult

ToolFn = Callable[..., ToolResult]


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: ToolFn

    def spec(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {tool.name: tool for tool in tools or []}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[dict[str, object]]:
        return [tool.spec() for tool in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name=name, content=f"unknown tool: {name}")
        required = [key for key in tool.parameters.get("required", []) if key not in arguments]
        if required:
            return ToolResult(name=name, content=f"missing arguments: {', '.join(required)}")
        allowed = set(tool.parameters.get("properties", {}))
        filtered = {key: value for key, value in arguments.items() if key in allowed}
        try:
            return tool.run(**filtered)
        except Exception as exc:  # a tool failure is an observation, not a crash
            return ToolResult(name=name, content=f"tool error: {type(exc).__name__}: {exc}")


def _format_hits(hits: list[Scored]) -> str:
    if not hits:
        return "no matching passages"
    lines = [
        f"- {scored.chunk.chunk_id} ({scored.chunk.location()}): "
        f"{' '.join(scored.chunk.text.split())[:240]}"
        for scored in hits
    ]
    return "\n".join(lines)


def build_default_tools(
    pipeline: RetrievalPipeline,
    vector_index: InMemoryVectorIndex,
) -> ToolRegistry:
    """The minimum useful toolset: search, fetch one chunk, list what is indexed."""

    def search_corpus(query: str, limit: int = 5) -> ToolResult:
        result = pipeline.run(query)
        hits = list(result.candidates)[:limit]
        return ToolResult(name="search_corpus", content=_format_hits(hits), chunks=tuple(hits))

    def fetch_chunk(chunk_id: str) -> ToolResult:
        chunk = vector_index.get(chunk_id)
        if chunk is None:
            return ToolResult(name="fetch_chunk", content=f"no such chunk: {chunk_id}")
        scored = Scored(chunk=chunk, score=1.0, retriever="fetch")
        return ToolResult(name="fetch_chunk", content=chunk.text, chunks=(scored,))

    def list_sources() -> ToolResult:
        sources = sorted({chunk.source for chunk in vector_index.all_chunks()})
        return ToolResult(name="list_sources", content="\n".join(f"- {s}" for s in sources))

    return ToolRegistry(
        [
            Tool(
                name="search_corpus",
                description="Search the indexed corpus for passages relevant to a query.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to look for."},
                        "limit": {"type": "integer", "description": "Max passages (1-10)."},
                    },
                    "required": ["query"],
                },
                run=search_corpus,
            ),
            Tool(
                name="fetch_chunk",
                description="Fetch the full text of one passage by its chunk id.",
                parameters={
                    "type": "object",
                    "properties": {"chunk_id": {"type": "string"}},
                    "required": ["chunk_id"],
                },
                run=fetch_chunk,
            ),
            Tool(
                name="list_sources",
                description="List the source documents currently indexed.",
                parameters={"type": "object", "properties": {}},
                run=list_sources,
            ),
        ]
    )
