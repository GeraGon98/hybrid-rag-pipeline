from __future__ import annotations

import json

import pytest

from contracts import RespuestaConCitas, StructuredOutputError, generate_structured
from contracts.structured_output import sanitize_schema
from core.errors import AllProvidersFailedError, PermanentProviderError
from core.llm_client import LLMClient
from tests.conftest import FakeProvider, make_response, read_events

MESSAGES = [
    {"role": "system", "content": "Answer with citations."},
    {"role": "user", "content": "Question: What is T-1042?"},
]
CHUNKS = {"RB-07#descripcion": "Ticket T-1042 is an inventory synchronization issue."}
VALID = {
    "encontrado": True,
    "respuesta": "It is an inventory synchronization issue.",
    "citas": [{"chunk_id": "RB-07#descripcion", "fragmento": "Ticket T-1042 is an"}],
}


def respond(payload) -> object:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return make_response("groq", text=text)


def make_client(script, logger, fake_sleep, **capabilities):
    provider = FakeProvider("groq", script, **capabilities)
    return provider, LLMClient([provider], logger, sleep=fake_sleep)


def test_valid_on_first_try(logger, fake_sleep):
    provider, client = make_client([respond(VALID)], logger, fake_sleep)

    result = generate_structured(
        client, MESSAGES, RespuestaConCitas, validation_context={"chunks": CHUNKS}
    )

    assert result.citas[0].chunk_id == "RB-07#descripcion"
    assert len(provider.calls) == 1


def test_invented_citation_is_repaired_with_context(logger, fake_sleep):
    invented = {**VALID, "citas": [{"chunk_id": "RB-99#x", "fragmento": "texto inventado"}]}
    provider, client = make_client([respond(invented), respond(VALID)], logger, fake_sleep)

    result = generate_structured(
        client, MESSAGES, RespuestaConCitas, validation_context={"chunks": CHUNKS}
    )

    assert result == RespuestaConCitas(**VALID)
    repair = provider.calls[1]["messages"][-1]["content"]
    assert "RB-99#x" in repair and "RB-07#descripcion" in repair


def test_invalid_json_is_repaired(logger, fake_sleep):
    provider, client = make_client(
        [respond('{"encontrado": true'), respond(VALID)], logger, fake_sleep
    )

    generate_structured(client, MESSAGES, RespuestaConCitas)

    second_history = provider.calls[1]["messages"]
    assert second_history[-2] == {"role": "assistant", "content": '{"encontrado": true'}
    assert "does not match the required format" in second_history[-1]["content"]


def test_repairs_exhausted_raise(logger, fake_sleep):
    bad = respond({**VALID, "citas": []})
    provider, client = make_client([bad, bad, bad], logger, fake_sleep)

    with pytest.raises(StructuredOutputError) as exc_info:
        generate_structured(client, MESSAGES, RespuestaConCitas, max_repairs=2)

    assert exc_info.value.attempts == 3
    assert "at least one citation" in exc_info.value.last_error
    assert len(provider.calls) == 3


def test_permanent_provider_error_is_not_repaired(logger, fake_sleep):
    error = PermanentProviderError("groq", "unsupported schema", 400)
    provider, client = make_client([error], logger, fake_sleep)

    with pytest.raises(AllProvidersFailedError):
        generate_structured(client, MESSAGES, RespuestaConCitas)
    assert len(provider.calls) == 1


def test_strategy_json_schema(logger, fake_sleep):
    provider, client = make_client([respond(VALID)], logger, fake_sleep)

    generate_structured(client, MESSAGES, RespuestaConCitas)

    params = provider.calls[0]["params"]
    assert params["response_format"]["type"] == "json_schema"
    assert params["response_format"]["json_schema"]["strict"] is True
    assert provider.calls[0]["messages"] == MESSAGES


def test_strategy_json_object_adds_schema_to_system(logger, fake_sleep):
    provider, client = make_client(
        [respond(VALID)], logger, fake_sleep, json_schema=False, json_object=True
    )

    generate_structured(client, MESSAGES, RespuestaConCitas)

    call = provider.calls[0]
    assert call["params"]["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in call["messages"][0]["content"]
    assert MESSAGES[0]["content"] == "Answer with citations."


def test_strategy_instruction_only_strips_fences(logger, fake_sleep):
    fenced = respond("```json\n" + json.dumps(VALID) + "\n```")
    provider, client = make_client(
        [fenced], logger, fake_sleep, json_schema=False, json_object=False
    )

    assert generate_structured(client, MESSAGES, RespuestaConCitas).encontrado is True
    assert "response_format" not in provider.calls[0]["params"]


def test_repair_attempt_and_schema_in_log(logger, log_path, fake_sleep):
    _, client = make_client([respond("not json"), respond(VALID)], logger, fake_sleep)

    generate_structured(
        client, MESSAGES, RespuestaConCitas, log_extra={"retrieval_strategy": "bm25"}
    )

    events = read_events(log_path)
    assert [e["repair_attempt"] for e in events] == [0, 1]
    assert all(e["structured_schema"] == "RespuestaConCitas" for e in events)
    assert all(e["retrieval_strategy"] == "bm25" for e in events)


def test_sanitize_resolves_refs_and_keeps_local_rules():
    sent = sanitize_schema(RespuestaConCitas.model_json_schema())

    dumped = json.dumps(sent)
    assert "$ref" not in dumped and "$defs" not in dumped
    assert "maxLength" not in dumped
    cita = sent["properties"]["citas"]["items"]
    assert cita["additionalProperties"] is False
    assert cita["required"] == ["chunk_id", "fragmento"]
    assert sent["required"] == list(RespuestaConCitas.model_fields)
