"""Embedders: EMBEDDING_DIM-dimensional vectors, always compared by cosine similarity (ADR-0005).

Unit length is not guaranteed: HashEmbedder returns the zero vector for text without tokens and
fastembed leaves the demo model's mean-pooled vectors unnormalized. Never use a raw dot product.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol

from frontdesk.db.models import EMBEDDING_DIM
from frontdesk.safety.normalize import tokens
from frontdesk.settings import EmbeddingBackend

DEMO_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    @property
    def identity(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Deterministic bag-of-tokens vectors for tests and CI.

    It says nothing about semantic quality: evaluations run with it only check
    contracts.
    """

    dimension = EMBEDDING_DIM
    identity = "hash-v1"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * EMBEDDING_DIM
        for token in tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIM
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class FastEmbedEmbedder:
    """Local multilingual model for the keyless demo (`uv sync --extra demo`)."""

    dimension = EMBEDDING_DIM

    def __init__(self, model_name: str = DEMO_MODEL) -> None:
        from fastembed import TextEmbedding

        self.identity = f"fastembed:{model_name}"
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(value) for value in vector] for vector in self._model.embed(list(texts))]


class NoEmbedder:
    """No embedder (Albi, 2026-10-16): a deployment whose knowledge rides in the prompt searches
    nothing (frontdesk.knowledge.in_prompt), and its image holds no model. Its versions are
    activated without vectors."""

    dimension = EMBEDDING_DIM
    identity = "none"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("this deployment has no embedder: its knowledge is in the prompt")


def build_embedder(backend: EmbeddingBackend) -> Embedder:
    if backend is EmbeddingBackend.HASH:
        return HashEmbedder()
    if backend is EmbeddingBackend.NONE:
        return NoEmbedder()
    return FastEmbedEmbedder()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
