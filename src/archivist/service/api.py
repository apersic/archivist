"""HTTP surface (optional, requires the `service` extra).

Three deliberate choices:

* The index is built once at startup and shared, because embedding the corpus
  per request is the most common way to make a RAG demo look slow.
* `/ask` returns citations as structured data, not just prose, so a client can
  render source links rather than regex the answer.
* `/ask/stream` streams server-sent events. Users tolerate latency they can
  watch; a 3 second silent wait reads as broken.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from archivist.app import Archivist, build
from archivist.config import Settings
from archivist.types import Answer

try:  # optional dependency, installed via the `service` extra
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse

    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    FASTAPI_AVAILABLE = False


def _serialise(answer: Answer) -> dict[str, Any]:
    return {
        "question": answer.question,
        "status": answer.status.value,
        "answer": answer.text,
        "citations": [
            {"marker": c.marker, "chunk_id": c.chunk_id, "location": c.location}
            for c in answer.citations
        ],
        "passages": [
            {
                "chunk_id": scored.chunk.chunk_id,
                "location": scored.chunk.location(),
                "score": round(scored.score, 4),
                "retriever": scored.retriever,
            }
            for scored in answer.contexts
        ],
        "usage": answer.usage,
    }


def create_app(settings: Settings | None = None, app_instance: Archivist | None = None) -> Any:
    if not FASTAPI_AVAILABLE:  # pragma: no cover - depends on the environment
        raise RuntimeError("install the 'service' extra to run the HTTP service")

    archivist = app_instance or build(settings or Settings.from_env())
    api = FastAPI(title="Archivist", version="0.1.0")

    @api.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "chunks": archivist.chunk_count,
            "embedder": archivist.embedder.name,
            "llm": archivist.llm.name,
        }

    @api.get("/sources")
    def sources() -> dict[str, Any]:
        documents = sorted({chunk.source for chunk in archivist.vector_index.all_chunks()})
        return {"count": len(documents), "sources": documents}

    @api.post("/ask")
    def ask(payload: dict[str, Any]) -> dict[str, Any]:
        question = str(payload.get("question", "")).strip()
        if not question:
            raise HTTPException(status_code=422, detail="question is required")
        mode = str(payload.get("mode", "plain"))
        history = [str(turn) for turn in payload.get("history", [])]
        if mode not in {"plain", "agentic"}:
            raise HTTPException(status_code=422, detail="mode must be 'plain' or 'agentic'")
        answer = (
            archivist.research(question, history=history)
            if mode == "agentic"
            else archivist.ask(question, history=history)
        )
        return _serialise(answer)

    @api.post("/ask/stream")
    def ask_stream(payload: dict[str, Any]) -> StreamingResponse:
        question = str(payload.get("question", "")).strip()
        if not question:
            raise HTTPException(status_code=422, detail="question is required")

        async def events() -> AsyncIterator[bytes]:
            # Retrieval finishes before generation starts, so the sources can be
            # shown while the answer is still being written.
            result = archivist.pipeline.run(question)
            yield _sse(
                "sources",
                {
                    "passages": [
                        {"marker": citation.marker, "location": citation.location}
                        for citation in result.context.citations
                    ]
                },
            )
            answer = archivist.answerer.answer(result.standalone_question, result.context)
            for word in answer.text.split(" "):
                yield _sse("token", {"text": word + " "})
            yield _sse("done", _serialise(answer))

        return StreamingResponse(events(), media_type="text/event-stream")

    return api


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()
