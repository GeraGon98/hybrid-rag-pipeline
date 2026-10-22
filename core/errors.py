"""Provider error hierarchy.

Transient errors are retried; permanent errors immediately advance to the next provider.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base error for any LLM provider."""

    def __init__(self, provider: str, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code

    def __str__(self) -> str:
        code = f" (HTTP {self.status_code})" if self.status_code is not None else ""
        return f"[{self.provider}]{code} {self.args[0]}"


class TransientProviderError(ProviderError):
    """Retryable error: 429, 500, 502, 503, 504, timeouts, and network failures."""


class PermanentProviderError(ProviderError):
    """Non-retryable error such as an invalid key or malformed request."""


class AllProvidersFailedError(Exception):
    """All providers failed, or none supports the requested capability."""

    def __init__(self, errors: list[ProviderError], reason: str | None = None) -> None:
        self.errors = errors
        detail = "; ".join(f"{type(e).__name__} {e}" for e in errors) or reason or "no providers"
        super().__init__(f"All providers failed: {detail}")
