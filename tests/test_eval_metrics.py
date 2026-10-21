from __future__ import annotations

import json
from pathlib import Path

import pytest

from retrieval.metrics import hit_at_k, mean, recall_at_k, reciprocal_rank

DATASET = Path(__file__).resolve().parents[1] / "data" / "eval_questions.jsonl"


def test_recall_counts_fraction_of_expected():
    assert recall_at_k(["a", "b", "c"], ["a", "z"], k=3) == 0.5
    assert recall_at_k(["a", "b", "c"], ["c"], k=2) == 0.0


def test_recall_requires_expected():
    with pytest.raises(ValueError):
        recall_at_k(["a"], [], k=1)


def test_hit_at_k():
    assert hit_at_k(["a", "b"], ["b", "z"], k=2) == 1.0
    assert hit_at_k(["a", "b"], ["b"], k=1) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x"], ["a"]) == 0.0


def test_mean_of_empty_is_zero():
    assert mean([]) == 0.0


def test_dataset_shape():
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 40
    assert {r["kind"] for r in rows} == {"lexical", "semantic", "mixed", "no_answer"}
    for row in rows:
        assert bool(row["expected_chunk_ids"]) == (row["kind"] != "no_answer")
        assert all("#" in cid for cid in row["expected_chunk_ids"])
