"""Construcción y carga del índice: Qdrant embebido + `chunks.jsonl` para BM25."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ingest.chunker import Chunk, chunk_documents
from ingest.loaders import load_directory
from retrieval.bm25 import BM25Index
from retrieval.embeddings import (
    CrossEncoderScorer,
    Embedder,
    PairScorer,
    SentenceTransformerEmbedder,
)
from retrieval.reranker import Reranker
from retrieval.retriever import Retriever
from retrieval.vector_store import VectorStore

if TYPE_CHECKING:
    from core.config import Settings

CHUNKS_FILE = "chunks.jsonl"
QDRANT_DIR = "qdrant"


def build_index(
    docs_dir: Path, settings: Settings, embedder: Embedder | None = None
) -> list[Chunk]:
    """Carga, trocea e indexa `docs_dir`. Reemplaza el índice anterior."""
    docs = load_directory(docs_dir)
    if not docs:
        raise FileNotFoundError(f"{docs_dir} no tiene documentos .md ni .pdf.")
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    save_chunks(chunks, settings.index_dir / CHUNKS_FILE)

    store = VectorStore(
        embedder or SentenceTransformerEmbedder(settings.embedding_model),
        settings.index_dir / QDRANT_DIR,
    )
    try:
        store.index(chunks)
    finally:
        store.close()
    return chunks


def load_retriever(
    settings: Settings,
    embedder: Embedder | None = None,
    scorer: PairScorer | None = None,
) -> Retriever:
    """Abre el índice existente. El cross-encoder se carga hasta que se usa."""
    chunks = load_chunks(settings.index_dir / CHUNKS_FILE)
    store = VectorStore(
        embedder or SentenceTransformerEmbedder(settings.embedding_model),
        settings.index_dir / QDRANT_DIR,
    )
    reranker = Reranker(scorer or CrossEncoderScorer(settings.reranker_model))
    return Retriever(
        chunks,
        store,
        BM25Index(chunks),
        reranker,
        candidates=settings.top_k_candidates,
        rrf_k=settings.rrf_k,
    )


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Corre `uv run main.py ingest` primero.")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Chunk.from_dict(json.loads(line)) for line in lines if line.strip()]
