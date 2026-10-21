"""Utilidades compartidas por los tests. Ningún test usa la red ni descarga modelos."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from core.llm_client import LLMResponse
from core.logger import CallLogger
from ingest.chunker import Chunk
from retrieval.bm25 import BM25Index, tokenize
from retrieval.reranker import Reranker
from retrieval.retriever import Retriever
from retrieval.vector_store import VectorStore


class FakeProvider:
    """Proveedor configurable: devuelve (o lanza) los elementos de `script` en orden."""

    def __init__(
        self,
        name: str,
        script: list,
        model: str = "fake-model",
        *,
        json_schema: bool = True,
        json_object: bool = True,
    ) -> None:
        self.name = name
        self.model = model
        self.supports_json_schema = json_schema
        self.supports_json_object = json_object
        self._script = list(script)
        self.calls: list[dict] = []

    def generate(self, messages: list[dict], **params) -> LLMResponse:
        self.calls.append({"messages": messages, "params": params})
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeSleep:
    """Reemplazo de time.sleep que solo registra las esperas."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


class FakeEmbedder:
    """Bolsa de palabras hasheada y normalizada: textos con palabras en común se parecen."""

    dim = 64

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in tokenize(text):
            vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class FakeScorer:
    """Fracción de tokens de la consulta presentes en el texto."""

    def __init__(self) -> None:
        self.calls = 0

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls += 1
        wanted = set(tokenize(query))
        return [len(wanted & set(tokenize(t))) / (len(wanted) or 1) for t in texts]


def make_response(provider: str = "fake", text: str = "ok", cost: float = 0.0001) -> LLMResponse:
    return LLMResponse(
        text=text,
        provider=provider,
        model="fake-model",
        tokens_in=10,
        tokens_out=5,
        latency_ms=12.0,
        cost_usd=cost,
    )


def read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def make_chunk(chunk_id: str, text: str, section: str = "Doc > Sección") -> Chunk:
    doc_id = chunk_id.split("#")[0]
    return Chunk(chunk_id, doc_id, f"{doc_id}.md", doc_id, section, "runbook", None, text)


SAMPLE_CHUNKS = [
    make_chunk("RB-07#descripcion", "El ticket T-1042 es la desincronización de inventario."),
    make_chunk("SLA-B#tiempos", "NetLink responde en 30 minutos ante una falla total del enlace."),
    make_chunk("POL-02#contrasenas", "Las contraseñas tienen mínimo 14 caracteres y MFA."),
    make_chunk("POL-06#robo", "Reporta la pérdida o robo de la laptop en menos de 4 horas."),
]


def make_retriever(chunks: list[Chunk] | None = None, scorer: FakeScorer | None = None):
    chunks = chunks or SAMPLE_CHUNKS
    store = VectorStore(FakeEmbedder())
    store.index(chunks)
    return Retriever(
        chunks, store, BM25Index(chunks), Reranker(scorer or FakeScorer()), candidates=4
    )


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "llm_calls.jsonl"


@pytest.fixture
def logger(log_path: Path) -> CallLogger:
    return CallLogger(log_path)


@pytest.fixture
def fake_sleep() -> FakeSleep:
    return FakeSleep()


@pytest.fixture
def retriever():
    r = make_retriever()
    yield r
    r.vector_store.close()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evita que las variables del entorno o el .env afecten los tests."""
    for var in ("GEMINI_API_KEY", "GROQ_API_KEY", "PRIMARY", "FALLBACKS", "MIN_RELEVANCE"):
        monkeypatch.delenv(var, raising=False)
