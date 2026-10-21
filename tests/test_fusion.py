from __future__ import annotations

import pytest

from retrieval.fusion import reciprocal_rank_fusion


def test_scores_follow_formula():
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_item_in_both_rankings_wins():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "d", "b"]], k=60)
    assert [doc for doc, _ in fused][:2] == ["c", "b"]


def test_ties_keep_first_appearance():
    fused = reciprocal_rank_fusion([["a"], ["b"]], k=60)
    assert [doc for doc, _ in fused] == ["a", "b"]


def test_empty_rankings():
    assert reciprocal_rank_fusion([[], []]) == []


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)
