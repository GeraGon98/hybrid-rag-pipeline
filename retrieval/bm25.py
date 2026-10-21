"""Índice y búsqueda léxica con BM25 (en memoria, se reconstruye desde chunks.jsonl)."""

from __future__ import annotations

import re
import unicodedata

from rank_bm25 import BM25Okapi

from ingest.chunker import Chunk

# Conserva códigos con guion o punto como un solo token: "T-1042", "GW-VPN-02", "99.7".
_TOKEN = re.compile(r"[a-z0-9]+(?:[-./_][a-z0-9]+)*")

STOPWORDS = frozenset(
    """
    a al algo ante antes como con contra cual cuales cuando cuanto cuanta cuantos cuantas de del
    desde donde durante e el ella ellas ellos en entre era es esa ese eso esta este esto estan
    fue fueron ha han hasta hay la las le les lo los mas me mi mis muy no nos o os otra otro para
    pero por porque que quien se sea segun ser si sin sobre son su sus tambien te tiene tienen
    todo todos tu un una uno unos unas y ya yo debo debe puedo hago hace cada
    """.split()
)


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return [t for t in _TOKEN.findall(normalized) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("BM25Index necesita al menos un chunk.")
        self.chunk_ids = [c.chunk_id for c in chunks]
        self._bm25 = BM25Okapi([tokenize(c.indexed_text) for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """`(chunk_id, score BM25)` de los `k` mejores con score positivo."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(self.chunk_ids, scores, strict=True), key=lambda p: p[1], reverse=True)
        return [(cid, float(score)) for cid, score in ranked[:k] if score > 0]
