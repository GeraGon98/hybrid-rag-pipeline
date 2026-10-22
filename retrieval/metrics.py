"""Retrieval metrics over chunk IDs."""

from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], expected: Sequence[str], k: int) -> float:
    """Fraction of expected chunks appearing in the first `k` results."""
    if not expected:
        raise ValueError("recall_at_k requires at least one expected chunk.")
    top = set(retrieved[:k])
    return sum(1 for cid in set(expected) if cid in top) / len(set(expected))


def hit_at_k(retrieved: Sequence[str], expected: Sequence[str], k: int) -> float:
    """1.0 si al menos un chunk esperado aparece en los primeros `k`."""
    return 1.0 if set(retrieved[:k]) & set(expected) else 0.0


def reciprocal_rank(retrieved: Sequence[str], expected: Sequence[str]) -> float:
    """Return `1 / rank` for the first expected chunk, or 0.0 if absent."""
    wanted = set(expected)
    for rank, cid in enumerate(retrieved, start=1):
        if cid in wanted:
            return 1.0 / rank
    return 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
