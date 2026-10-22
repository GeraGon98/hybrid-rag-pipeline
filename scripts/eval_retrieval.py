"""Compare retrieval strategies on `data/eval_questions.jsonl`.

Print Markdown tables ready for the README:
1. recall@k, hit@k, and MRR by strategy, globally and by question type.
2. Relevance-threshold calibration for answerable and unanswerable questions.
3. With `--answers`, citation fidelity for LLM-generated answers (uses real API calls).

Usage:
    uv run scripts/eval_retrieval.py --strategies vector bm25 hybrid hybrid_rerank --k 5
    uv run scripts/eval_retrieval.py --answers --answer-strategy hybrid_rerank
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from contracts.structured_output import StructuredOutputError  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.errors import AllProvidersFailedError  # noqa: E402
from core.llm_client import LLMClient  # noqa: E402
from generation.rag import RAGAssistant  # noqa: E402
from retrieval.index import load_retriever  # noqa: E402
from retrieval.metrics import hit_at_k, mean, recall_at_k, reciprocal_rank  # noqa: E402
from retrieval.retriever import STRATEGIES, Retriever  # noqa: E402

DEFAULT_DATASET = ROOT / "data" / "eval_questions.jsonl"
KINDS = ("lexical", "semantic", "mixed")


def load_questions(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def evaluate_strategy(
    retriever: Retriever,
    questions: list[dict],
    strategy: str,
    k: int,
    thresholds: dict,
    verbose: bool = False,
) -> dict:
    recalls: dict[str, list[float]] = defaultdict(list)
    hits, rrs, latencies = [], [], []
    relevance = {"answerable": [], "no_answer": []}
    gate = {"answerable_passed": 0, "no_answer_rejected": 0}

    for q in questions:
        start = time.perf_counter()
        result = retriever.search(q["question"], strategy, k)
        latencies.append((time.perf_counter() - start) * 1000)
        passed = result.relevance >= thresholds[result.relevance_source]
        if verbose:
            rr = reciprocal_rank(result.chunk_ids, q["expected_chunk_ids"])
            print(
                f"  {strategy:<13} {q['id']} {q['kind']:<9} relevance={result.relevance:.4f} "
                f"rr={rr:.2f} top={result.chunk_ids[:1]}"
            )
        if not q["expected_chunk_ids"]:
            relevance["no_answer"].append(result.relevance)
            gate["no_answer_rejected"] += not passed
            continue
        relevance["answerable"].append(result.relevance)
        gate["answerable_passed"] += passed
        recall = recall_at_k(result.chunk_ids, q["expected_chunk_ids"], k)
        recalls["all"].append(recall)
        recalls[q["kind"]].append(recall)
        hits.append(hit_at_k(result.chunk_ids, q["expected_chunk_ids"], k))
        rrs.append(reciprocal_rank(result.chunk_ids, q["expected_chunk_ids"]))

    return {
        "strategy": strategy,
        "recall": mean(recalls["all"]),
        "recall_by_kind": {kind: mean(recalls[kind]) for kind in KINDS},
        "hit": mean(hits),
        "mrr": mean(rrs),
        "latency_ms": mean(latencies),
        "relevance": {key: mean(values) for key, values in relevance.items()},
        "relevance_range": {
            key: (min(values), max(values)) if values else (0.0, 0.0)
            for key, values in relevance.items()
        },
        "answerable": len(relevance["answerable"]),
        "no_answer": len(relevance["no_answer"]),
        **gate,
    }


def print_retrieval_table(rows: list[dict], k: int) -> None:
    print(
        f"\n| Strategy | recall@{k} | hit@{k} | MRR | lexical recall | semantic recall "
        f"| mixed recall | avg latency |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        by_kind = r["recall_by_kind"]
        print(
            f"| `{r['strategy']}` | {r['recall']:.3f} | {r['hit']:.3f} | {r['mrr']:.3f} "
            f"| {by_kind['lexical']:.3f} | {by_kind['semantic']:.3f} | {by_kind['mixed']:.3f} "
            f"| {r['latency_ms']:.0f} ms |"
        )


def print_gate_table(rows: list[dict], thresholds: dict) -> None:
    print(
        f"\nThresholds: MIN_RELEVANCE={thresholds['reranker']} (reranker), "
        f"MIN_VECTOR_SIMILARITY={thresholds['vector']} (cosine)\n"
    )
    print(
        "| Strategy | answerable relevance (min-max) | unanswerable relevance (min-max) "
        "| answerable passed | unanswerable rejected |"
    )
    print("|---|---|---|---|---|")
    for r in rows:
        lo_a, hi_a = r["relevance_range"]["answerable"]
        lo_n, hi_n = r["relevance_range"]["no_answer"]
        print(
            f"| `{r['strategy']}` | {r['relevance']['answerable']:.3f} ({lo_a:.3f}-{hi_a:.3f}) "
            f"| {r['relevance']['no_answer']:.3f} ({lo_n:.3f}-{hi_n:.3f}) "
            f"| {r['answerable_passed']}/{r['answerable']} "
            f"| {r['no_answer_rejected']}/{r['no_answer']} |"
        )


def evaluate_answers(assistant: RAGAssistant, questions: list[dict], strategy: str) -> dict:
    counts = defaultdict(int)
    cost = 0.0
    for q in questions:
        expected = set(q["expected_chunk_ids"])
        key = "answerable" if expected else "no_answer"
        counts[key] += 1
        try:
            result = assistant.ask(q["question"], strategy)
        except (AllProvidersFailedError, StructuredOutputError) as e:
            counts["errors"] += 1
            print(f"  {q['id']} error: {e}", file=sys.stderr)
            continue
        cost += result.usage.cost_usd
        answer = result.answer
        cited = {c.chunk_id for c in answer.citas}
        if expected:
            counts["answered"] += answer.encontrado
            counts["cites_expected"] += bool(cited & expected)
        else:
            counts["rejected"] += not answer.encontrado
        status = "answered" if answer.encontrado else "no information"
        print(f"  {q['id']} {status:<15} citations={sorted(cited) or '-'}")
    return {**counts, "cost_usd": cost}


def print_answer_table(stats: dict, strategy: str) -> None:
    answerable, no_answer = stats.get("answerable", 0), stats.get("no_answer", 0)
    print(f"\nAnswer fidelity with `{strategy}` (all citations passed validation):\n")
    print("| Metric | Value |")
    print("|---|---|")
    print(f"| Answerable questions answered | {stats.get('answered', 0)}/{answerable} |")
    print(f"| ...citing an expected chunk | {stats.get('cites_expected', 0)}/{answerable} |")
    print(f"| Unanswerable questions rejected | {stats.get('rejected', 0)}/{no_answer} |")
    print(f"| Errors (provider or exhausted validation) | {stats.get('errors', 0)} |")
    print(f"| Total cost | ${stats['cost_usd']:.5f} |")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--answers", action="store_true", help="Also evaluate generated answers.")
    parser.add_argument("--answer-strategy", choices=STRATEGIES, default="hybrid_rerank")
    parser.add_argument("--verbose", action="store_true", help="Show per-question details.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    questions = load_questions(args.dataset)
    retriever = load_retriever(settings)
    thresholds = {"reranker": settings.min_relevance, "vector": settings.min_vector_similarity}
    try:
        retriever.search(
            questions[0]["question"], "vector", args.k
        )  # Load the model before measuring.
        rows = [
            evaluate_strategy(retriever, questions, strategy, args.k, thresholds, args.verbose)
            for strategy in args.strategies
        ]
        answerable = sum(1 for q in questions if q["expected_chunk_ids"])
        print(f"{len(questions)} questions ({answerable} answerable) · k={args.k}")
        print_retrieval_table(rows, args.k)
        print_gate_table(rows, thresholds)

        if args.answers:
            print(f"\nGenerating answers with {args.answer_strategy}...")
            client = LLMClient.from_settings(settings)
            assistant = RAGAssistant.from_settings(settings, retriever, client)
            stats = evaluate_answers(assistant, questions, args.answer_strategy)
            print_answer_table(stats, args.answer_strategy)
    finally:
        retriever.vector_store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
