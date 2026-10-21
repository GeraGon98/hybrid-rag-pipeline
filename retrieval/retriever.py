"""Recuperación con cuatro estrategias comparables.

- `vector`: top-k por similitud coseno.
- `bm25`: top-k léxico.
- `hybrid`: `top_k_candidates` de cada buscador fusionados con RRF.
- `hybrid_rerank`: los candidatos fusionados se reordenan con el cross-encoder.

`relevance` resume qué tan bueno es el mejor resultado para decidir si vale la pena llamar
al LLM. Los scores BM25 y RRF no están calibrados, así que se usa el score del reranker
(0-1) cuando existe y, si no, la similitud coseno del mejor resultado vectorial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ingest.chunker import Chunk
from retrieval.bm25 import BM25Index
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import Reranker
from retrieval.vector_store import VectorStore

STRATEGIES = ("vector", "bm25", "hybrid", "hybrid_rerank")
Strategy = Literal["vector", "bm25", "hybrid", "hybrid_rerank"]


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    strategy: str
    chunks: list[ScoredChunk]
    relevance: float
    relevance_source: Literal["reranker", "vector"]

    @property
    def chunk_ids(self) -> list[str]:
        return [sc.chunk.chunk_id for sc in self.chunks]


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        vector_store: VectorStore,
        bm25: BM25Index,
        reranker: Reranker | None = None,
        *,
        candidates: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.chunks = {c.chunk_id: c for c in chunks}
        self.vector_store = vector_store
        self.bm25 = bm25
        self.reranker = reranker
        self.candidates = candidates
        self.rrf_k = rrf_k

    def search(self, question: str, strategy: str = "hybrid", k: int = 5) -> RetrievalResult:
        if strategy not in STRATEGIES:
            raise ValueError(f"Estrategia desconocida: {strategy!r}. Usa una de: {STRATEGIES}.")
        if strategy == "bm25":
            ranked = self.bm25.search(question, k)
            vector_hits = self.vector_store.search(question, 1)
            return self._result(strategy, ranked, _top_score(vector_hits), "vector")

        vector_hits = self.vector_store.search(question, max(k, self.candidates))
        if strategy == "vector":
            return self._result(strategy, vector_hits[:k], _top_score(vector_hits), "vector")

        lexical_hits = self.bm25.search(question, self.candidates)
        fused = reciprocal_rank_fusion(
            [[cid for cid, _ in vector_hits], [cid for cid, _ in lexical_hits]], k=self.rrf_k
        )
        if strategy == "hybrid":
            return self._result(strategy, fused[:k], _top_score(vector_hits), "vector")

        if self.reranker is None:
            raise ValueError("La estrategia hybrid_rerank necesita un reranker.")
        candidates = [self.chunks[cid] for cid, _ in fused[: self.candidates]]
        reranked = self.reranker.rerank(question, candidates, k)
        scored = [ScoredChunk(chunk, score) for chunk, score in reranked]
        relevance = scored[0].score if scored else 0.0
        return RetrievalResult(strategy, scored, relevance, "reranker")

    def _result(
        self,
        strategy: str,
        ranked: list[tuple[str, float]],
        relevance: float,
        source: Literal["reranker", "vector"],
    ) -> RetrievalResult:
        scored = [ScoredChunk(self.chunks[cid], score) for cid, score in ranked]
        return RetrievalResult(strategy, scored, relevance, source)


def _top_score(hits: list[tuple[str, float]]) -> float:
    return hits[0][1] if hits else 0.0
