"""CLI del pipeline RAG híbrido.

Uso:
    uv run main.py ingest [data/docs]
    uv run main.py ask "¿Cuál es el SLA del proveedor de enlaces?"
    uv run main.py ask --strategy bm25 "¿Qué es el ticket T-1042?"
    uv run main.py ask --provider ollama --k 3 "..."
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from pydantic import ValidationError  # noqa: E402

from contracts.structured_output import StructuredOutputError  # noqa: E402
from core.config import KNOWN_PROVIDERS, Settings, get_settings  # noqa: E402
from core.errors import AllProvidersFailedError  # noqa: E402
from core.llm_client import LLMClient  # noqa: E402
from generation.rag import RAGAnswer, RAGAssistant  # noqa: E402
from retrieval.index import build_index, load_retriever  # noqa: E402
from retrieval.retriever import STRATEGIES  # noqa: E402

GRAY = "\033[90m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pregunta a la documentación de operaciones con búsqueda híbrida y citas."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="Indexa una carpeta de documentos .md/.pdf.")
    ingest.add_argument("path", type=Path, nargs="?", help="Carpeta (por defecto DOCS_DIR).")

    ask = commands.add_parser("ask", help="Responde una pregunta con citas.")
    ask.add_argument("question", help="Pregunta en lenguaje natural.")
    ask.add_argument("--strategy", choices=STRATEGIES, default="hybrid_rerank")
    ask.add_argument("--k", type=int, help="Chunks que llegan al LLM (por defecto TOP_K).")
    ask.add_argument(
        "--provider", choices=KNOWN_PROVIDERS, help="Fuerza un proveedor y desactiva fallbacks."
    )
    ask.add_argument("--prompt-version", type=int, metavar="N", help="Versión de rag_answer.")
    return parser.parse_args(argv)


def build_settings(provider: str | None = None) -> Settings:
    if provider:
        return Settings(primary=provider, fallbacks=[])
    return get_settings()


def fail(message: str) -> int:
    print(f"Error: {message}", file=sys.stderr)
    return 1


def run_ingest(args: argparse.Namespace, settings: Settings) -> int:
    docs_dir = args.path or settings.docs_dir
    try:
        chunks = build_index(docs_dir, settings)
    except FileNotFoundError as e:
        return fail(str(e))
    docs = len({c.doc_id for c in chunks})
    print(f"{GREEN}OK{RESET} {docs} documentos · {len(chunks)} chunks · {settings.index_dir}")
    return 0


def print_answer(result: RAGAnswer) -> None:
    answer, retrieval = result.answer, result.retrieval
    color = GREEN if answer.encontrado else YELLOW
    print(f"{color}{answer.respuesta}{RESET}")
    for cita in answer.citas:
        print(f"  [{cita.chunk_id}] «{cita.fragmento}»")
    usage = result.usage
    llm = (
        f"{usage.calls} llamadas · {usage.tokens_in}/{usage.tokens_out} tokens · "
        f"{usage.latency_ms:.0f} ms · ${usage.cost_usd:.6f}"
        if result.used_llm
        else "sin llamada al LLM"
    )
    print(
        f"{GRAY}  {retrieval.strategy} · relevancia {retrieval.relevance:.3f} "
        f"({retrieval.relevance_source}, umbral {result.threshold:.2f}) · "
        f"chunks {', '.join(retrieval.chunk_ids) or '-'} · {llm}{RESET}"
    )


def run_ask(args: argparse.Namespace, settings: Settings) -> int:
    try:
        retriever = load_retriever(settings)
    except FileNotFoundError as e:
        return fail(str(e))
    client = LLMClient.from_settings(settings)
    assistant = RAGAssistant.from_settings(settings, retriever, client, args.prompt_version)
    try:
        result = assistant.ask(args.question, args.strategy, args.k)
    except (AllProvidersFailedError, StructuredOutputError) as e:
        return fail(str(e))
    finally:
        retriever.vector_store.close()
    print_answer(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = build_settings(getattr(args, "provider", None))
    except ValidationError as e:
        details = "; ".join(err["msg"].removeprefix("Value error, ") for err in e.errors())
        return fail(f"configuración inválida. {details}")
    if args.command == "ingest":
        return run_ingest(args, settings)
    if args.command == "ask":
        return run_ask(args, settings)
    return fail(f"comando desconocido: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
