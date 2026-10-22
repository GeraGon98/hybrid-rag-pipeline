"""LLM provider adapters.

All providers use OpenAI-compatible APIs through `httpx`, without SDKs. Each adapter
only communicates with its provider; `LLMClient` manages routing, retries, and fallback.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol

import httpx

from core import pricing
from core.errors import PermanentProviderError, TransientProviderError
from core.llm_client import LLMResponse

if TYPE_CHECKING:
    from pydantic import SecretStr

    from core.config import Settings

TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


class Provider(Protocol):
    name: str

    def generate(self, messages: list[dict], **params) -> LLMResponse: ...


class OpenAICompatibleProvider:
    """Base adapter for any OpenAI-compatible `/chat/completions` API.

    Capabilities:
    - `supports_json_schema`: accepts `response_format={"type": "json_schema", ...}`.
    - `supports_json_object`: accepts `response_format={"type": "json_object"}`.
    Base defaults match Groq with `openai/gpt-oss-*`.
    """

    name: str = ""
    base_url: str = ""
    supports_json_schema: bool = True
    supports_json_object: bool = True

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 60,
        *,
        name: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.name = name or self.name
        self.base_url = (base_url or self.base_url).rstrip("/")
        self.model = model
        self._api_key = api_key
        self.timeout_s = timeout_s

    def generate(self, messages: list[dict], **params) -> LLMResponse:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {"model": self.model, "messages": messages, **params}

        start = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout_s,
            )
        except httpx.TimeoutException as e:
            raise TransientProviderError(self.name, f"timeout: {e}") from e
        except httpx.TransportError as e:
            raise TransientProviderError(self.name, f"connection error: {e}") from e
        latency_ms = (time.perf_counter() - start) * 1000

        if response.status_code in TRANSIENT_STATUS_CODES:
            raise TransientProviderError(self.name, _short(response), response.status_code)
        if response.status_code >= 400:
            raise PermanentProviderError(self.name, _short(response), response.status_code)

        try:
            data = response.json()
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise PermanentProviderError(
                self.name, f"unexpected response: {e!r}", response.status_code
            ) from e

        usage = data.get("usage") or {}
        tokens_in = int(usage.get("prompt_tokens") or 0)
        tokens_out = int(usage.get("completion_tokens") or 0)
        return LLMResponse(
            text=text,
            provider=self.name,
            model=data.get("model") or self.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=pricing.cost_usd(self.model, tokens_in, tokens_out, provider=self.name),
            finish_reason=choice.get("finish_reason"),
        )


def _short(response: httpx.Response, limit: int = 200) -> str:
    """Short error-body summary that never includes request headers."""
    return " ".join(response.text.split())[:limit] or response.reason_phrase


class GeminiProvider(OpenAICompatibleProvider):
    # Official OpenAI-compatible endpoint (https://ai.google.dev/gemini-api/docs/openai).
    name = "gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    supports_json_schema = True
    supports_json_object = False


class OllamaProvider(OpenAICompatibleProvider):
    # docs.ollama.com/api/openai-compatibility: JSON mode is supported; schema mode is unconfirmed.
    name = "ollama"
    base_url = "http://localhost:11434/v1"
    supports_json_schema = False
    supports_json_object = True


class GroqProvider(OpenAICompatibleProvider):
    name = "groq"
    base_url = "https://api.groq.com/openai/v1"


def build_provider(name: str, settings: Settings) -> Provider:
    """Build a provider adapter from configuration."""

    def key(secret: SecretStr | None) -> str | None:
        return secret.get_secret_value() if secret is not None else None

    timeout = settings.timeout_s
    if name == "gemini":
        return GeminiProvider(settings.gemini_model, key(settings.gemini_api_key), timeout)
    if name == "groq":
        return GroqProvider(settings.groq_model, key(settings.groq_api_key), timeout)
    if name == "ollama":
        return OllamaProvider(
            settings.ollama_model,
            None,
            settings.ollama_timeout_s,
            base_url=settings.ollama_base_url,
        )
    raise ValueError(f"Unknown provider: {name!r}")
