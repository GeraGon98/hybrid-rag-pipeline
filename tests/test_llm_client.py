from __future__ import annotations

import pytest

from core.errors import AllProvidersFailedError, PermanentProviderError, TransientProviderError
from core.llm_client import LLMClient, prompt_chars
from tests.conftest import FakeProvider, make_response, read_events

MESSAGES = [{"role": "user", "content": "hola"}]


def test_transient_error_is_retried(logger, fake_sleep):
    provider = FakeProvider("groq", [TransientProviderError("groq", "429"), make_response()])
    client = LLMClient([provider], logger, sleep=fake_sleep)

    assert client.generate(MESSAGES).text == "ok"
    assert len(provider.calls) == 2
    assert len(fake_sleep.waits) == 1


def test_permanent_error_falls_back(logger, log_path, fake_sleep):
    primary = FakeProvider("groq", [PermanentProviderError("groq", "401", 401)])
    backup = FakeProvider("gemini", [make_response("gemini")])
    client = LLMClient([primary, backup], logger, sleep=fake_sleep)

    assert client.generate(MESSAGES).provider == "gemini"
    events = read_events(log_path)
    assert [(e["provider"], e["success"], e["fallback"]) for e in events] == [
        ("groq", False, False),
        ("gemini", True, True),
    ]


def test_retries_exhausted_then_fallback(logger, fake_sleep):
    errors = [TransientProviderError("groq", "503")] * 3
    primary = FakeProvider("groq", errors)
    backup = FakeProvider("ollama", [make_response("ollama")])
    client = LLMClient([primary, backup], logger, max_retries=2, sleep=fake_sleep)

    assert client.generate(MESSAGES).provider == "ollama"
    assert len(primary.calls) == 3
    assert len(fake_sleep.waits) == 2


def test_all_fail(logger, fake_sleep):
    error = PermanentProviderError("groq", "400", 400)
    client = LLMClient([FakeProvider("groq", [error])], logger, sleep=fake_sleep)
    with pytest.raises(AllProvidersFailedError) as exc_info:
        client.generate(MESSAGES)
    assert exc_info.value.errors == [error]


def test_log_extra_includes_retrieval_fields(logger, log_path, fake_sleep):
    client = LLMClient([FakeProvider("groq", [make_response()])], logger, sleep=fake_sleep)

    client.generate(
        MESSAGES, log_extra={"retrieval_strategy": "hybrid", "chunk_ids": ["RB-01#sintomas"]}
    )

    (event,) = read_events(log_path)
    assert event["retrieval_strategy"] == "hybrid"
    assert event["chunk_ids"] == ["RB-01#sintomas"]
    assert event["prompt_chars"] == 4
    assert "hola" not in log_path.read_text(encoding="utf-8")


def test_unknown_log_field_is_rejected(logger, fake_sleep):
    client = LLMClient([FakeProvider("groq", [make_response()])], logger, sleep=fake_sleep)
    with pytest.raises(TypeError):
        client.generate(MESSAGES, log_extra={"prompt_text": "secreto"})


def test_prompt_chars_counts_all_messages():
    assert prompt_chars([{"role": "system", "content": "abc"}, *MESSAGES]) == 7


def test_totals_accumulate_and_reset(logger, fake_sleep):
    responses = [make_response(cost=0.001), make_response(cost=0.002)]
    client = LLMClient([FakeProvider("groq", responses)], logger, sleep=fake_sleep)

    client.generate(MESSAGES)
    client.generate(MESSAGES)
    totals = client.reset_totals()

    assert totals.calls == 2
    assert totals.tokens_in == 20
    assert totals.cost_usd == pytest.approx(0.003)
    assert client.totals.calls == 0
