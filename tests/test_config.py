from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.config import Settings


def make(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_defaults_only_need_ollama():
    settings = make(primary="ollama", fallbacks=[])
    assert settings.top_k == 5
    assert settings.rrf_k == 60


def test_missing_key_names_env_var():
    with pytest.raises(ValidationError, match="GROQ_API_KEY"):
        make(primary="groq", fallbacks=[])


def test_unknown_provider():
    with pytest.raises(ValidationError, match="Unknown provider"):
        make(primary="openai", fallbacks=[])


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        make(primary="ollama", fallbacks=[], chunk_size=100, chunk_overlap=100)


def test_top_k_not_above_candidates():
    with pytest.raises(ValidationError, match="TOP_K"):
        make(primary="ollama", fallbacks=[], top_k=30, top_k_candidates=20)
