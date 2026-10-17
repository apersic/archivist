"""Minimal tracing and usage accounting.

A RAG answer that is wrong is usually wrong in one specific stage. The trace
records every stage with its timing and its inputs/outputs summary, which is
what makes the failure attributable instead of mysterious. The shape mirrors
OpenTelemetry spans so this can be exported later without touching callers.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    name: str
    span_id: str
    parent_id: str | None
    started_at: float
    duration_ms: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    spans: list[Span] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    _stack: list[str] = field(default_factory=list)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        span = Span(
            name=name,
            span_id=uuid.uuid4().hex[:8],
            parent_id=self._stack[-1] if self._stack else None,
            started_at=time.perf_counter(),
            attributes=dict(attributes),
        )
        self.spans.append(span)
        self._stack.append(span.span_id)
        try:
            yield span
        except Exception as exc:
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.duration_ms = (time.perf_counter() - span.started_at) * 1000
            self._stack.pop()

    def record_usage(self, usage: dict[str, int]) -> None:
        for key, value in usage.items():
            self.usage[key] = self.usage.get(key, 0) + value

    def render(self) -> str:
        depth_of: dict[str | None, int] = {None: 0}
        lines: list[str] = []
        for span in self.spans:
            depth = depth_of.get(span.parent_id, 0)
            depth_of[span.span_id] = depth + 1
            attrs = " ".join(f"{k}={v}" for k, v in span.attributes.items())
            timing = f"{span.duration_ms:.1f}ms" if span.duration_ms is not None else "running"
            marker = "x" if span.error else "-"
            lines.append(f"{'  ' * depth}{marker} {span.name} ({timing}) {attrs}".rstrip())
            if span.error:
                lines.append(f"{'  ' * (depth + 1)}! {span.error}")
        if self.usage:
            lines.append("usage: " + ", ".join(f"{k}={v}" for k, v in sorted(self.usage.items())))
        return "\n".join(lines)
