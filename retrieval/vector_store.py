"""Vector indexing and search with embedded, serverless Qdrant."""

from __future__ import annotations

import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ingest.chunker import Chunk
from retrieval.embeddings import Embedder

COLLECTION = "ops_docs"
_ID_NAMESPACE = uuid.UUID("6f1c2a52-6d1e-4c39-9a51-6b0f3c2d9e10")


class VectorStore:
    """`path=None` uses an in-memory collection, which is useful in tests."""

    def __init__(
        self, embedder: Embedder, path: Path | None = None, collection: str = COLLECTION
    ) -> None:
        self.embedder = embedder
        self.collection = collection
        if path is None:
            self.client = QdrantClient(location=":memory:")
        else:
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))

    def index(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        """Replace the entire collection with the supplied chunks."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
        )
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.embedder.embed_documents([c.indexed_text for c in batch])
            points = [
                PointStruct(id=_point_id(c.chunk_id), vector=v, payload=c.to_dict())
                for c, v in zip(batch, vectors, strict=True)
            ]
            self.client.upsert(self.collection, points=points)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return `(chunk_id, cosine_similarity)` for the `k` nearest chunks."""
        if not self.client.collection_exists(self.collection):
            raise FileNotFoundError("No vector index found. Run `main.py ingest` first.")
        result = self.client.query_points(
            self.collection, query=self.embedder.embed_query(query), limit=k, with_payload=True
        )
        return [(point.payload["chunk_id"], float(point.score)) for point in result.points]

    def close(self) -> None:
        self.client.close()


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))
