"""Embedding and reranking models behind small protocols.

`sentence-transformers` models load on first use rather than at import time, so
`ask --strategy bm25` avoids model startup and tests can use fakes.
"""

from __future__ import annotations

import math
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class PairScorer(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance of each text to the query, in the 0-1 range."""
        ...


class SentenceTransformerEmbedder:
    """Normalized embeddings (cosine similarity equals dot product).

    E5 models expect `query: ` and `passage: ` prefixes; omitting them reduces quality.
    """

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._e5 = "e5" in model_name.lower()

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        model = self.model
        getter = getattr(model, "get_embedding_dimension", None)
        return int(getter() if getter else model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts] if self._e5 else texts
        return self._encode(prefixed)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"query: {text}" if self._e5 else text])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return [vector.tolist() for vector in vectors]


class CrossEncoderScorer:
    def __init__(self, model_name: str, batch_size: int = 16) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        raw = self.model.predict(
            [(query, text) for text in texts], batch_size=self.batch_size, show_progress_bar=False
        )
        scores = [float(value) for value in raw]
        if any(s < 0 or s > 1 for s in scores):  # The model returned logits.
            scores = [1 / (1 + math.exp(-s)) for s in scores]
        return scores
