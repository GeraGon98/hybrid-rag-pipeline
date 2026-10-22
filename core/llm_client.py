"""LLM client with retries, exponential backoff, and provider fallback."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.errors import AllProvidersFailedError, ProviderError, TransientProviderError

if TYPE_CHECKING:
    from core.config import Settings
    from core.logger import CallLogger
    from core.providers import Provider


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_usd: float
    finish_reason: str | None = None


@dataclass
class UsageTotals:
    """Successful-response totals since the last `reset_totals()`."""

    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0

    def add(self, response: LLMResponse) -> None:
        self.calls += 1
        self.tokens_in += response.tokens_in
        self.tokens_out += response.tokens_out
        self.cost_usd += response.cost_usd
        self.latency_ms += response.latency_ms


def prompt_chars(messages: list[dict]) -> int:
    return sum(len(str(message.get("content") or "")) for message in messages)


class LLMClient:
    def __init__(
        self,
        providers: list[Provider],
        logger: CallLogger,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not providers:
            raise ValueError("LLMClient requires at least one provider.")
        self.providers = providers
        self.logger = logger
        self.max_retries = max_retries
        self._sleep = sleep
        self.totals = UsageTotals()

    @classmethod
    def from_settings(cls, settings: Settings) -> LLMClient:
        """Build the configured primary-plus-fallback provider chain."""
        from core.logger import CallLogger
        from core.providers import build_provider

        names = [settings.primary, *settings.fallbacks]
        providers = [build_provider(name, settings) for name in names]
        return cls(providers, CallLogger(settings.log_path), settings.max_retries)

    def reset_totals(self) -> UsageTotals:
        """Return the current totals and start a new accumulator."""
        previous, self.totals = self.totals, UsageTotals()
        return previous

    def generate(
        self,
        messages: list[dict],
        *,
        log_extra: dict | None = None,
        response_format: dict | None = None,
        **params,
    ) -> LLMResponse:
        """Request a response by trying providers in order.

        - `TransientProviderError`: retry with 1 s -> 2 s -> 4 s backoff plus jitter.
        - `PermanentProviderError`: skip retries and advance to the next provider.
        - Si todos fallan: `AllProvidersFailedError`.
        """
        chars = prompt_chars(messages)
        extra = dict(log_extra or {})
        if response_format is not None:
            params["response_format"] = response_format
        errors: list[ProviderError] = []

        for index, provider in enumerate(self.providers):
            is_fallback = index > 0
            for attempt in range(self.max_retries + 1):
                start = time.perf_counter()
                try:
                    response = provider.generate(messages, **params)
                except ProviderError as e:
                    self._log_failure(provider, e, start, is_fallback, chars, extra)
                    if isinstance(e, TransientProviderError) and attempt < self.max_retries:
                        self._sleep(2**attempt + random.uniform(0, 0.25))
                        continue
                    errors.append(e)
                    break  # Permanent failure or retries exhausted: next provider.
                self.logger.log(
                    provider=response.provider,
                    model=response.model,
                    tokens_in=response.tokens_in,
                    tokens_out=response.tokens_out,
                    latency_ms=response.latency_ms,
                    cost_usd=response.cost_usd,
                    fallback=is_fallback,
                    success=True,
                    error_type=None,
                    prompt_chars=chars,
                    **extra,
                )
                self.totals.add(response)
                return response

        raise AllProvidersFailedError(errors)

    def _log_failure(
        self,
        provider: Provider,
        error: ProviderError,
        start: float,
        is_fallback: bool,
        chars: int,
        extra: dict,
    ) -> None:
        self.logger.log(
            provider=provider.name,
            model=getattr(provider, "model", None),
            tokens_in=0,
            tokens_out=0,
            latency_ms=(time.perf_counter() - start) * 1000,
            cost_usd=0.0,
            fallback=is_fallback,
            success=False,
            error_type=type(error).__name__,
            prompt_chars=chars,
            **extra,
        )
