from __future__ import annotations

from typing import Any

import pytest

from archivist.app import Archivist

fastapi = pytest.importorskip("fastapi", reason="requires the 'service' extra")

from fastapi.testclient import TestClient  # noqa: E402

from archivist.service.api import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client(app: Archivist) -> Any:
    return TestClient(create_app(app_instance=app))


def test_healthz_reports_what_is_loaded(client: Any) -> None:
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["chunks"] > 0


def test_ask_returns_structured_citations(client: Any) -> None:
    question = "What is the retention period for events_raw?"
    response = client.post("/ask", json={"question": question})
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "answered"
    assert "30 days" in body["answer"]
    assert body["citations"]
    assert body["citations"][0]["location"]
    assert body["passages"][0]["retriever"]


def test_agentic_mode_is_selectable(client: Any) -> None:
    response = client.post(
        "/ask", json={"question": "Which job writes events_curated?", "mode": "agentic"}
    )
    assert response.status_code == 200
    assert response.json()["status"] in {"answered", "abstained"}


def test_bad_requests_are_rejected(client: Any) -> None:
    assert client.post("/ask", json={"question": "  "}).status_code == 422
    assert client.post("/ask", json={"question": "hi", "mode": "wat"}).status_code == 422


def test_stream_emits_sources_before_tokens(client: Any) -> None:
    with client.stream(
        "POST", "/ask/stream", json={"question": "What is the p95 latency budget?"}
    ) as response:
        body = "".join(chunk for chunk in response.iter_text())

    assert response.status_code == 200
    assert body.index("event: sources") < body.index("event: token") < body.index("event: done")
