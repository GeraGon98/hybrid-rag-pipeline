from __future__ import annotations

import json

from core.llm_client import LLMClient
from generation.rag import NO_INFO, RAGAssistant
from tests.conftest import FakeProvider, make_response, read_events

VALID = {
    "encontrado": True,
    "respuesta": "It is an inventory synchronization issue.",
    "citas": [{"chunk_id": "RB-07#descripcion", "fragmento": "Ticket T-1042 is an"}],
}


def make_assistant(retriever, script, logger, fake_sleep, **kwargs):
    provider = FakeProvider("groq", script)
    client = LLMClient([provider], logger, sleep=fake_sleep)
    return provider, RAGAssistant(retriever, client, **kwargs)


def respond(payload: dict):
    return make_response("groq", json.dumps(payload, ensure_ascii=False))


def test_answer_with_validated_citations(retriever, logger, log_path, fake_sleep):
    provider, assistant = make_assistant(
        retriever, [respond(VALID)], logger, fake_sleep, min_relevance=0.1
    )

    result = assistant.ask("What is ticket T-1042?", "hybrid_rerank", k=2)

    assert result.used_llm is True
    assert result.answer.citas[0].chunk_id == "RB-07#descripcion"
    assert result.usage.calls == 1
    user = provider.calls[0]["messages"][-1]["content"]
    assert "<documents>" in user and "[RB-07#descripcion]" in user
    assert user.endswith("Question: What is ticket T-1042?")
    (event,) = read_events(log_path)
    assert event["retrieval_strategy"] == "hybrid_rerank"
    assert event["prompt_name"] == "rag_answer"
    assert "RB-07#descripcion" in event["chunk_ids"]


def test_low_relevance_skips_llm(retriever, logger, log_path, fake_sleep):
    provider, assistant = make_assistant(retriever, [], logger, fake_sleep, min_relevance=0.5)

    result = assistant.ask("How many vacation days do I have?", "hybrid_rerank")

    assert result.used_llm is False
    assert result.answer.encontrado is False
    assert result.answer.respuesta == NO_INFO
    assert provider.calls == []
    assert not log_path.exists()


def test_vector_threshold_applies_without_reranker(retriever, logger, fake_sleep):
    provider, assistant = make_assistant(
        retriever, [], logger, fake_sleep, min_vector_similarity=0.99
    )

    result = assistant.ask("What is ticket T-1042?", "hybrid")

    assert result.threshold == 0.99
    assert result.used_llm is False


def test_invented_citation_is_repaired(retriever, logger, fake_sleep):
    invented = {**VALID, "citas": [{"chunk_id": "RB-99#x", "fragmento": "invented"}]}
    provider, assistant = make_assistant(
        retriever, [respond(invented), respond(VALID)], logger, fake_sleep, min_relevance=0.1
    )

    result = assistant.ask("What is ticket T-1042?", "hybrid_rerank", k=2)

    assert result.answer.citas[0].chunk_id == "RB-07#descripcion"
    assert len(provider.calls) == 2
    assert result.usage.calls == 2


def test_context_budget_hides_low_ranked_chunks(retriever, logger, fake_sleep):
    budget = {"system": 1200, "documents": 40, "user": 100}
    provider, assistant = make_assistant(
        retriever, [respond(VALID)], logger, fake_sleep, min_relevance=0.1, context_budget=budget
    )

    result = assistant.ask("What is ticket T-1042?", "hybrid_rerank", k=4)

    user = provider.calls[0]["messages"][-1]["content"]
    assert "[RB-07#descripcion]" in user
    assert user.count("[") < 4
    assert "documents" in result.context_report.truncated_sections
