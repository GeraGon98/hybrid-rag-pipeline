"""Asistente RAG: recupera, decide si hay contexto suficiente y genera la respuesta citada."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from context.manager import ContextManager, ContextReport
from contracts.answer import RespuestaConCitas
from contracts.structured_output import generate_structured
from core.llm_client import LLMClient, UsageTotals
from prompting.promptkit import PromptKit
from retrieval.retriever import RetrievalResult, Retriever

if TYPE_CHECKING:
    from core.config import Settings

PROMPT_NAME = "rag_answer"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
NO_INFO = "No encontré información sobre eso en la documentación disponible."


@dataclass
class RAGAnswer:
    question: str
    answer: RespuestaConCitas
    retrieval: RetrievalResult
    threshold: float
    used_llm: bool
    usage: UsageTotals
    context_report: ContextReport | None = None
    prompt_version: int | None = None


class RAGAssistant:
    def __init__(
        self,
        retriever: Retriever,
        client: LLMClient,
        *,
        prompts: PromptKit | None = None,
        prompt_version: int | None = None,
        top_k: int = 5,
        min_relevance: float = 0.01,
        min_vector_similarity: float = 0.80,
        max_repairs: int = 2,
        context_budget: dict[str, int] | None = None,
    ) -> None:
        self.retriever = retriever
        self.client = client
        self.prompt = (prompts or PromptKit(PROMPTS_DIR)).get(PROMPT_NAME, prompt_version)
        self.top_k = top_k
        self.thresholds = {"reranker": min_relevance, "vector": min_vector_similarity}
        self.max_repairs = max_repairs
        self.context_budget = context_budget or {"system": 1200, "documents": 3000, "user": 300}

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        retriever: Retriever,
        client: LLMClient,
        prompt_version: int | None = None,
    ) -> RAGAssistant:
        return cls(
            retriever,
            client,
            prompt_version=prompt_version,
            top_k=settings.top_k,
            min_relevance=settings.min_relevance,
            min_vector_similarity=settings.min_vector_similarity,
            max_repairs=settings.max_repairs,
            context_budget=settings.context_budget,
        )

    def ask(self, question: str, strategy: str = "hybrid", k: int | None = None) -> RAGAnswer:
        """Responde `question`. Si la relevancia queda bajo el umbral, no llama al LLM."""
        self.client.reset_totals()
        retrieval = self.retriever.search(question, strategy, k or self.top_k)
        threshold = self.thresholds[retrieval.relevance_source]

        if not retrieval.chunks or retrieval.relevance < threshold:
            answer = RespuestaConCitas(encontrado=False, respuesta=NO_INFO, citas=[])
            return RAGAnswer(
                question, answer, retrieval, threshold, False, self.client.reset_totals()
            )

        messages, report, visible = self._messages(question, retrieval)
        answer = generate_structured(
            self.client,
            messages,
            RespuestaConCitas,
            self.max_repairs,
            log_extra={
                "prompt_name": self.prompt.name,
                "prompt_version": self.prompt.version,
                "retrieval_strategy": retrieval.strategy,
                "chunk_ids": list(visible),
            },
            validation_context={"chunks": visible},
            **self.prompt.model_params,
        )
        return RAGAnswer(
            question,
            answer,
            retrieval,
            threshold,
            True,
            self.client.reset_totals(),
            report,
            self.prompt.version,
        )

    def _messages(
        self, question: str, retrieval: RetrievalResult
    ) -> tuple[list[dict], ContextReport | None, dict[str, str]]:
        """Arma los mensajes con presupuesto y regresa los chunks que quedaron visibles."""
        system, user = (m["content"] for m in self.prompt.render(question=question))
        manager = ContextManager(self.context_budget)
        manager.add("system", system)
        total = len(retrieval.chunks)
        for rank, scored in enumerate(retrieval.chunks):
            chunk = scored.chunk
            manager.add("documents", f"[{chunk.chunk_id}] {chunk.indexed_text}", total - rank)
        manager.add("user", user)
        messages = manager.build()

        user_content = messages[-1]["content"]
        visible = {
            sc.chunk.chunk_id: sc.chunk.indexed_text
            for sc in retrieval.chunks
            if f"[{sc.chunk.chunk_id}]" in user_content
        }
        return messages, manager.last_report, visible
