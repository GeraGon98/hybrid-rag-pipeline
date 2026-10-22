from __future__ import annotations

import pytest

from context import ContextManager, estimate_tokens
from context.manager import TRUNCATION_MARK


def words(text: str) -> int:
    """Trivial test counter: one word equals one token."""
    return len(text.split())


BUDGET = {"system": 10, "examples": 10, "history": 6, "documents": 6, "user": 5, "output": 20}


def make(budget: dict | None = None) -> ContextManager:
    return ContextManager(budget or BUDGET, count_tokens=words)


def test_estimate_tokens_is_chars_over_four():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_respects_budgets_when_everything_fits():
    cm = make()
    cm.add("system", "you are classifier")
    cm.add("history", "one two three")
    cm.add("user", "short question")
    cm.build()
    report = cm.last_report
    assert report.tokens_by_section == {"system": 3, "history": 3, "user": 2}
    assert report.truncated_sections == []
    assert report.total_tokens == 8


def test_drops_lowest_priority_first_then_oldest():
    cm = make()
    cm.add("history", "a1 a2", priority=0)  # Oldest, low priority.
    cm.add("history", "b1 b2", priority=2)  # Important.
    cm.add("history", "c1 c2", priority=0)
    cm.add("history", "d1 d2", priority=1)
    messages = cm.build()  # Eight words, budget six: one item is dropped.

    content = messages[0]["content"]
    assert "a1" not in content
    assert all(word in content for word in ("b1", "c1", "d1"))
    assert cm.last_report.tokens_by_section["history"] == 6
    assert cm.last_report.truncated_sections == ["history"]


def test_keeps_high_priority_even_if_oldest():
    cm = make({"history": 4})
    cm.add("history", "old important", priority=5)
    cm.add("history", "new one", priority=0)
    cm.add("history", "new two", priority=0)
    content = cm.build()[0]["content"]
    assert "old important" in content
    assert "new two" in content
    assert "new one" not in content


def test_truncates_single_item_with_mark():
    cm = make()
    cm.add("documents", "one two three four five six seven eight nine ten")
    content = cm.build()[0]["content"]
    assert "one two three four five" in content
    assert "six" not in content
    assert TRUNCATION_MARK in content
    assert cm.last_report.tokens_by_section["documents"] <= BUDGET["documents"]
    assert cm.last_report.truncated_sections == ["documents"]


def test_system_over_budget_raises():
    cm = make({"system": 3, "user": 5})
    cm.add("system", "this system is far too long")
    with pytest.raises(ValueError, match="never truncated"):
        cm.build()


def test_last_report_is_none_before_build():
    assert make().last_report is None


def test_output_order_and_tags():
    cm = make()
    cm.add("user", "the question")
    cm.add("documents", "a document")
    cm.add("history", "one turn")
    cm.add("examples", "an example")
    cm.add("system", "the system")
    system, user = cm.build()

    assert system["role"] == "system"
    assert system["content"].index("the system") < system["content"].index("<examples>")
    assert user["role"] == "user"
    body = user["content"]
    assert body.index("<history>") < body.index("<documents>") < body.index("the question")
    assert body.endswith("the question")


def test_unknown_section_or_missing_budget():
    cm = make({"system": 10})
    with pytest.raises(ValueError, match="Unknown section"):
        cm.add("notes", "x")
    with pytest.raises(ValueError, match="no assigned budget"):
        cm.add("history", "x")
