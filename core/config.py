"""Pipeline configuration loaded from environment variables and `.env`.

Missing variables fail at startup rather than midway through a request.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings

KNOWN_PROVIDERS = ("gemini", "groq", "ollama")

# Provider -> API-key environment variable. Local Ollama requires no key.
_KEY_ENV_VARS = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "extra": "ignore"}

    gemini_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    groq_model: str = "openai/gpt-oss-120b"
    ollama_model: str = "qwen2.5:7b"
    ollama_base_url: str = "http://localhost:11434/v1"
    primary: str = "groq"
    fallbacks: list[str] = ["ollama"]
    timeout_s: float = 60
    ollama_timeout_s: float = 300
    max_retries: int = 3
    max_repairs: int = 2
    log_path: Path = Path("logs/llm_calls.jsonl")

    docs_dir: Path = Path("data/docs")
    index_dir: Path = Path("data/index")
    chunk_size: int = 900
    chunk_overlap: int = 150

    embedding_model: str = "intfloat/multilingual-e5-small"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    top_k_candidates: int = 20
    top_k: int = 5
    rrf_k: int = 60
    # "No information found" thresholds: reranker score (0-1) for hybrid_rerank and
    # cosine similarity of the best vector result for other strategies.
    min_relevance: float = 0.01
    min_vector_similarity: float = 0.80

    # Token budget per context section (see context/manager.py).
    context_budget: dict[str, int] = {"system": 1200, "documents": 3000, "user": 300, "output": 800}

    @model_validator(mode="after")
    def _check(self) -> Settings:
        for name in [self.primary, *self.fallbacks]:
            if name not in KNOWN_PROVIDERS:
                raise ValueError(
                    f"Unknown provider: {name!r}. Use one of: {', '.join(KNOWN_PROVIDERS)}."
                )
            if name == "ollama":
                continue
            if getattr(self, f"{name}_api_key") is None:
                env_var = _KEY_ENV_VARS[name]
                raise ValueError(
                    f"Missing {name} API key. Add {env_var}=... to your .env file "
                    f"or remove {name!r} from PRIMARY/FALLBACKS."
                )
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be >= 0 and less than CHUNK_SIZE.")
        if self.top_k > self.top_k_candidates:
            raise ValueError("TOP_K cannot exceed TOP_K_CANDIDATES.")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return configuration, created on first use rather than at import time."""
    return Settings()
