"""Report LLM calls from logs/llm_calls.jsonl, broken down by retrieval strategy.

Usage:
    uv run scripts/report.py [path/to/file.jsonl]
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_PATH = Path("logs/llm_calls.jsonl")


def load_events(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        sys.exit("No calls have been logged yet. Run main.py ask first.")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def percentiles(values: list[float]) -> tuple[float, float]:
    """Return p50 and p95; quantiles() requires at least two data points."""
    if len(values) == 1:
        return values[0], values[0]
    q = statistics.quantiles(values, n=100, method="inclusive")
    return q[49], q[94]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else DEFAULT_PATH

    events = load_events(path)
    ok = [e for e in events if e["success"]]
    total_cost = sum(e["cost_usd"] for e in events)
    print(f"Log events:            {len(events)}  ({len(ok)} responses)")
    print(f"Total cost:            ${total_cost:.5f}")
    if ok:
        p50, p95 = percentiles([e["latency_ms"] for e in ok])
        fallbacks = sum(e["fallback"] for e in ok)
        repairs = sum(1 for e in ok if (e.get("repair_attempt") or 0) > 0)
        print(f"Latency p50 / p95:     {p50:,.0f} ms / {p95:,.0f} ms")
        print(
            f"Fallback:              {fallbacks / len(ok) * 100:.1f} %  ({fallbacks} of {len(ok)})"
        )
        print(f"Repair responses:      {repairs} of {len(ok)}")
    else:
        print("No successful responses; latency and fallback cannot be calculated.")

    counts = Counter((e["provider"], e["success"]) for e in events)
    print(f"\n{'Provider':<10}  {'success':>7}  {'errors':>7}")
    for provider in sorted({e["provider"] for e in events}):
        print(f"{provider:<10}  {counts[(provider, True)]:>6}  {counts[(provider, False)]:>7}")

    by_strategy: dict[str, list[dict]] = defaultdict(list)
    for e in ok:
        by_strategy[e.get("retrieval_strategy") or "-"].append(e)
    header = f"{'Strategy':<14}  {'calls':>8}  {'avg input tokens':>16}  {'cost':>10}"
    print(f"\n{header}  {'p50 ms':>8}")
    for strategy in sorted(by_strategy):
        items = by_strategy[strategy]
        p50, _ = percentiles([e["latency_ms"] for e in items])
        tokens = sum(e["tokens_in"] for e in items) / len(items)
        cost = sum(e["cost_usd"] for e in items)
        print(f"{strategy:<14}  {len(items):>8}  {tokens:>14,.0f}  ${cost:>9.5f}  {p50:>8,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
