"""Reciprocal Rank Fusion (Cormack et al., 2009)."""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Fusiona rankings de ids: `score(d) = sum(1 / (k + rank))`, con rank desde 1.

    Solo usa posiciones, no scores, así que mezcla sin calibrar coseno y BM25. Los empates
    se resuelven por la primera aparición en el orden de `rankings`.
    """
    if k <= 0:
        raise ValueError("k debe ser positivo.")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
