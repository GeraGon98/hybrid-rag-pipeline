from __future__ import annotations

import pytest

from core.config import Settings
from retrieval.index import build_index, load_chunks, load_retriever
from retrieval.retriever import Retriever
from retrieval.vector_store import VectorStore
from tests.conftest import SAMPLE_CHUNKS, FakeEmbedder, FakeScorer, make_retriever


@pytest.mark.parametrize("strategy", ["vector", "bm25", "hybrid", "hybrid_rerank"])
def test_every_strategy_finds_the_code(retriever, strategy):
    result = retriever.search("What is ticket T-1042?", strategy, k=2)

    assert result.strategy == strategy
    assert result.chunk_ids[0] == "RB-07#descripcion"
    assert len(result.chunks) <= 2


def test_relevance_source_depends_on_strategy(retriever):
    assert retriever.search("T-1042", "hybrid", 2).relevance_source == "vector"
    reranked = retriever.search("T-1042", "hybrid_rerank", 2)
    assert reranked.relevance_source == "reranker"
    assert reranked.relevance == reranked.chunks[0].score


def test_reranker_only_runs_for_hybrid_rerank():
    scorer = FakeScorer()
    r = make_retriever(scorer=scorer)
    try:
        for strategy in ("vector", "bm25", "hybrid"):
            r.search("T-1042", strategy, 2)
        assert scorer.calls == 0
        r.search("T-1042", "hybrid_rerank", 2)
        assert scorer.calls == 1
    finally:
        r.vector_store.close()


def test_unknown_strategy(retriever):
    with pytest.raises(ValueError, match="Unknown strategy"):
        retriever.search("x", "magic", 2)


def test_hybrid_rerank_requires_reranker():
    store = VectorStore(FakeEmbedder())
    store.index(SAMPLE_CHUNKS)
    from retrieval.bm25 import BM25Index

    r = Retriever(SAMPLE_CHUNKS, store, BM25Index(SAMPLE_CHUNKS), reranker=None)
    with pytest.raises(ValueError, match="reranker"):
        r.search("T-1042", "hybrid_rerank", 2)
    store.close()


def test_vector_store_reindex_replaces_collection():
    store = VectorStore(FakeEmbedder())
    store.index(SAMPLE_CHUNKS)
    store.index(SAMPLE_CHUNKS[:1])
    assert [cid for cid, _ in store.search("passwords", 10)] == ["RB-07#descripcion"]
    store.close()


def test_build_and_load_index_on_disk(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "RB-07.md").write_text(
        "---\nid: RB-07\ntitle: Inventory\n---\n\n# RB-07\n\n## Description\n\n"
        "Ticket T-1042 remains open.\n\n## Prevention\n\nValidate SKU length.\n",
        encoding="utf-8",
    )
    settings = Settings(_env_file=None, primary="ollama", fallbacks=[], index_dir=tmp_path / "idx")

    chunks = build_index(docs, settings, embedder=FakeEmbedder())
    assert [c.chunk_id for c in load_chunks(tmp_path / "idx" / "chunks.jsonl")] == [
        c.chunk_id for c in chunks
    ]

    retriever = load_retriever(settings, embedder=FakeEmbedder(), scorer=FakeScorer())
    try:
        assert retriever.search("T-1042", "hybrid_rerank", 1).chunk_ids == ["RB-07#description"]
    finally:
        retriever.vector_store.close()


def test_load_without_index_explains(tmp_path):
    settings = Settings(_env_file=None, primary="ollama", fallbacks=[], index_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="ingest"):
        load_retriever(settings, embedder=FakeEmbedder(), scorer=FakeScorer())
