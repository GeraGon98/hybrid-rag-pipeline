"""In-memory BM25 lexical index rebuilt from chunks.jsonl."""

from __future__ import annotations

import re
import unicodedata

from rank_bm25 import BM25Okapi

from ingest.chunker import Chunk

# Preserve hyphenated and dotted codes as one token: "T-1042", "GW-VPN-02", "99.7".
_TOKEN = re.compile(r"[a-z0-9]+(?:[-./_][a-z0-9]+)*")

STOPWORDS = frozenset(
    """
    a an and are as at be been before being between both but by can could did do does doing
    down during each few for from further had has have having he her here hers herself him
    himself his how i if in into is it its itself me more most my myself no nor not of off on
    once only or other our ours ourselves out over own same she should so some such than that
    the their theirs them themselves then there these they this those through to too under until
    up very was we were what when where which while who whom why will with you your yours yourself
    yourselves
    """.split()
)


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return [t for t in _TOKEN.findall(normalized) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("BM25Index requires at least one chunk.")
        self.chunk_ids = [c.chunk_id for c in chunks]
        self._bm25 = BM25Okapi([tokenize(c.indexed_text) for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return `(chunk_id, BM25_score)` for the top `k` positive-scoring chunks."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(self.chunk_ids, scores, strict=True), key=lambda p: p[1], reverse=True)
        return [(cid, float(score)) for cid, score in ranked[:k] if score > 0]
