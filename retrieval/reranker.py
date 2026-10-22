"""Candidate reranking with a cross-encoder."""

from __future__ import annotations

from ingest.chunker import Chunk
from retrieval.embeddings import PairScorer


class Reranker:
    def __init__(self, scorer: PairScorer) -> None:
        self.scorer = scorer

    def rerank(self, query: str, chunks: list[Chunk], k: int) -> list[tuple[Chunk, float]]:
        """Return the `k` most relevant chunks with cross-encoder scores in 0-1."""
        scores = self.scorer.score(query, [c.indexed_text for c in chunks])
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda p: p[1], reverse=True)
        return ranked[:k]
