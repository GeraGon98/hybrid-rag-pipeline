from __future__ import annotations

import pytest

from retrieval.bm25 import BM25Index, tokenize
from tests.conftest import SAMPLE_CHUNKS


def test_tokenize_keeps_codes_and_removes_accents_and_stopwords():
    assert tokenize("What is ticket T-1042 for GW-VPN-02?") == ["ticket", "t-1042", "gw-vpn-02"]
    assert tokenize("Resolution at 99.7%") == ["resolution", "99.7"]


def test_code_query_finds_exact_chunk():
    index = BM25Index(SAMPLE_CHUNKS)
    (top, score), *_ = index.search("T-1042", k=3)
    assert top == "RB-07#descripcion"
    assert score > 0


def test_only_positive_scores_are_returned():
    index = BM25Index(SAMPLE_CHUNKS)
    assert index.search("kubernetes", k=3) == []
    assert index.search("what is?", k=3) == []


def test_empty_index_rejected():
    with pytest.raises(ValueError):
        BM25Index([])
