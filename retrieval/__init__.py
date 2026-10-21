"""Recuperación: índice vectorial (Qdrant), BM25, fusión RRF y reranking con cross-encoder."""

from retrieval.fusion import reciprocal_rank_fusion
from retrieval.retriever import STRATEGIES, RetrievalResult, Retriever, ScoredChunk

__all__ = ["STRATEGIES", "RetrievalResult", "Retriever", "ScoredChunk", "reciprocal_rank_fusion"]
