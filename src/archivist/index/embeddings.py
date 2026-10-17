"""Embedding adapters.

`HashingEmbedder` is the default: a deterministic, dependency-free feature-hash
of character n-grams and word unigrams. It is weak at paraphrase but strong at
morphology, it never calls the network, and it makes the whole pipeline and its
evals reproducible on a laptop or in CI.

`OpenAIEmbedder` is the production path. Both satisfy `ports.Embedder`, so
switching is a config change and a re-index, and the eval harness tells you
what that change bought you.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from archivist.ports import Embedder
from archivist.text import tokenize

try:  # optional dependency, installed via the `openai` extra
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only without the extra
    OpenAI = None

def _bucket(token: str, dimensions: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    sign = 1.0 if value & 1 else -1.0
    return value % dimensions, sign


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


class HashingEmbedder:
    """Deterministic bag-of-features embedder with sublinear term weighting."""

    def __init__(self, dimensions: int = 512, char_ngrams: int = 4) -> None:
        self._dimensions = dimensions
        self._char_ngrams = char_ngrams

    @property
    def name(self) -> str:
        return f"hashing-{self._dimensions}d"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _features(self, text: str) -> dict[str, float]:
        counts: dict[str, float] = {}
        words = tokenize(text)
        for word in words:
            counts[word] = counts.get(word, 0.0) + 1.0
            if len(word) > self._char_ngrams:
                padded = f"^{word}$"
                for i in range(len(padded) - self._char_ngrams + 1):
                    gram = padded[i : i + self._char_ngrams]
                    counts[f"#{gram}"] = counts.get(f"#{gram}", 0.0) + 0.5
        for left, right in zip(words, words[1:], strict=False):
            bigram = f"{left}_{right}"
            counts[bigram] = counts.get(bigram, 0.0) + 0.75
        return counts

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token, count in self._features(text).items():
            index, sign = _bucket(token, self._dimensions)
            vector[index] += sign * (1.0 + math.log(count))
        return _normalise(vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OpenAIEmbedder:
    """Adapter for OpenAI embedding models (requires the `openai` extra)."""

    def __init__(self, model: str = "text-embedding-3-small", client: Any | None = None) -> None:
        self._model = model
        self._client = client
        self._dimensions = 1536 if model.endswith("small") else 3072

    @property
    def name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _ensure_client(self) -> Any:
        if self._client is None:
            if OpenAI is None:
                raise RuntimeError("install the 'openai' extra to use OpenAIEmbedder")
            self._client = OpenAI()
        return self._client

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._ensure_client().embeddings.create(model=self._model, input=texts)
        return [list(item.embedding) for item in response.data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 128):
            vectors.extend(self._embed_batch(texts[start : start + 128]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]


def build_embedder(name: str, dimensions: int = 512, model: str | None = None) -> Embedder:
    match name:
        case "hashing":
            return HashingEmbedder(dimensions=dimensions)
        case "openai":
            return OpenAIEmbedder(model=model or "text-embedding-3-small")
        case _:
            raise ValueError(f"unknown embedder: {name!r} (expected 'hashing' or 'openai')")
