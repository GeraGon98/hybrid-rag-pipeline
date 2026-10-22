"""Per-call cost calculation from input and output token counts."""

from __future__ import annotations

import warnings

# Illustrative prices; verify each provider's current rates before making decisions.
# Format: model ID -> (USD per 1M input tokens, USD per 1M output tokens).
PRICES_PER_1M: dict[str, tuple[float, float]] = {
    # Gemini (https://ai.google.dev/gemini-api/docs/pricing, paid tier)
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    # Groq (https://console.groq.com/docs/models)
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
}

_warned_models: set[str] = set()


def cost_usd(model: str, tokens_in: int, tokens_out: int, provider: str | None = None) -> float:
    """Return the USD cost of one call.

    Local Ollama calls always cost 0.0. Unknown models return 0.0 and emit one warning.
    """
    if provider == "ollama":
        return 0.0
    prices = PRICES_PER_1M.get(model)
    if prices is None:
        if model not in _warned_models:
            _warned_models.add(model)
            warnings.warn(
                f"Model has no price in core/pricing.py: {model!r}. Recording cost 0.0.",
                stacklevel=2,
            )
        return 0.0
    price_in, price_out = prices
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000
