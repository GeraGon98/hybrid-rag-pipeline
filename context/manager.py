"""Build context under per-section token budgets.

Output order: `system` and `examples` go in the system message; `history`, `documents`,
and `user` go in the user message, each in its tag, with the question last.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

SECTIONS = ("system", "examples", "history", "documents", "user")
RESERVED_SECTIONS = ("output",)  # Reserved in the budget but never populated.
TRUNCATION_MARK = "[…truncated]"

_TAGS = {"examples": "examples", "history": "history", "documents": "documents"}


def estimate_tokens(text: str) -> int:
    """Estimate tokens as `ceil(characters / 4)`.

    This character-based estimate allocates the budget despite provider tokenization
    differences. Billing uses the API's reported `usage`.
    """
    return math.ceil(len(text) / 4)


@dataclass
class ContextReport:
    tokens_by_section: dict[str, int]
    truncated_sections: list[str]
    total_tokens: int


@dataclass
class _Item:
    content: str
    priority: int
    order: int  # Arrival order: lower means older.
    tokens: int = field(default=0)


class ContextManager:
    def __init__(
        self, budget: dict[str, int], count_tokens: Callable[[str], int] = estimate_tokens
    ) -> None:
        unknown = set(budget) - set(SECTIONS) - set(RESERVED_SECTIONS)
        if unknown:
            raise ValueError(f"Unknown sections in budget: {sorted(unknown)}")
        self.budget = dict(budget)
        self.count_tokens = count_tokens
        self._items: dict[str, list[_Item]] = {section: [] for section in SECTIONS}
        self._counter = 0
        self._last_report: ContextReport | None = None

    def add(self, section: str, content: str, priority: int = 0) -> None:
        """Add an item. Higher `priority` values are discarded later."""
        if section not in SECTIONS:
            raise ValueError(f"Unknown section: {section!r}. Use one of: {SECTIONS}.")
        if section not in self.budget:
            raise ValueError(f"Section {section!r} has no assigned budget.")
        self._items[section].append(_Item(content, priority, self._counter))
        self._counter += 1

    @property
    def last_report(self) -> ContextReport | None:
        return self._last_report

    def build(self) -> list[dict]:
        """Apply budgets and return the `system` and `user` messages."""
        kept: dict[str, list[str]] = {}
        tokens_by_section: dict[str, int] = {}
        truncated: list[str] = []

        for section in SECTIONS:
            items = self._items[section]
            if not items:
                continue
            for item in items:
                item.tokens = self.count_tokens(item.content)
            budget = self.budget[section]
            if section == "system":
                used = sum(item.tokens for item in items)
                if used > budget:
                    raise ValueError(
                        f"The system message uses {used} tokens but its budget is {budget}. "
                        "It is never truncated; shorten it or increase its budget."
                    )
                fitted, was_cut = [item.content for item in items], False
            else:
                fitted, was_cut = self._fit(items, budget)
            kept[section] = fitted
            tokens_by_section[section] = sum(self.count_tokens(text) for text in fitted)
            if was_cut:
                truncated.append(section)

        self._last_report = ContextReport(
            tokens_by_section=tokens_by_section,
            truncated_sections=truncated,
            total_tokens=sum(tokens_by_section.values()),
        )
        return self._messages(kept)

    def _fit(self, items: list[_Item], budget: int) -> tuple[list[str], bool]:
        """Discard by priority and age, then truncate the final item if needed."""
        remaining = list(items)
        was_cut = False
        while len(remaining) > 1 and sum(i.tokens for i in remaining) > budget:
            victim = min(remaining, key=lambda i: (i.priority, i.order))
            remaining.remove(victim)
            was_cut = True

        texts = [item.content for item in sorted(remaining, key=lambda i: i.order)]
        if remaining and remaining[0].tokens > budget:  # One item remains and does not fit.
            cut = self._truncate(remaining[0].content, budget)
            texts = [cut] if cut else []
            was_cut = True
        return texts, was_cut

    def _truncate(self, text: str, budget: int) -> str | None:
        """Keep the longest prefix that fits alongside the truncation marker."""

        def fits(length: int) -> bool:
            candidate = f"{text[:length].rstrip()} {TRUNCATION_MARK}"
            return self.count_tokens(candidate) <= budget

        if not fits(0):
            return None
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if fits(mid):
                low = mid
            else:
                high = mid - 1
        return f"{text[:low].rstrip()} {TRUNCATION_MARK}"

    @staticmethod
    def _messages(kept: dict[str, list[str]]) -> list[dict]:
        system_parts = list(kept.get("system", []))
        if kept.get("examples"):
            system_parts.append(_wrap("examples", kept["examples"]))

        user_parts = [_wrap(s, kept[s]) for s in ("history", "documents") if kept.get(s)]
        user_parts.extend(kept.get("user", []))  # Keep the question last.

        messages = []
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})
        if user_parts:
            messages.append({"role": "user", "content": "\n\n".join(user_parts)})
        return messages


def _wrap(section: str, texts: list[str]) -> str:
    tag = _TAGS[section]
    body = "\n\n".join(texts)
    return f"<{tag}>\n{body}\n</{tag}>"
