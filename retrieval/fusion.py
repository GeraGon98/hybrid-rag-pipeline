"""Reciprocal Rank Fusion (Cormack et al., 2009)."""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse ID rankings with `score(d) = sum(1 / (k + rank))`, where rank starts at 1.

    Uses ranks rather than scores, allowing cosine and BM25 results to be combined without
    calibration. Ties are resolved by first appearance in `rankings`.
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
