"""Modelos de embeddings y de reranking detrás de protocolos pequeños.

Los modelos de `sentence-transformers` se cargan la primera vez que se usan, no al
importar: así `ask --strategy bm25` no paga la carga de modelos y los tests usan fakes.
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
        """Relevancia de cada texto para la consulta, en 0-1."""
        ...


class SentenceTransformerEmbedder:
    """Embeddings normalizados (coseno = producto punto).

    Los modelos E5 esperan los prefijos `query: ` y `passage: `; sin ellos la calidad cae.
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
        if any(s < 0 or s > 1 for s in scores):  # El modelo regresó logits.
            scores = [1 / (1 + math.exp(-s)) for s in scores]
        return scores
