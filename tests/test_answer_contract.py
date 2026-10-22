from __future__ import annotations

import pytest
from pydantic import ValidationError

from contracts.answer import RespuestaConCitas, normalize

CHUNKS = {
    "SLA-B#tiempos": "Total outage: response within 30 minutes and resolution within 4 hours.",
    "SLA-B#disponibilidad": "Committed monthly availability of 99.7% per circuit.",
}


def answer(**overrides) -> dict:
    data = {
        "encontrado": True,
        "respuesta": "NetLink responds within 30 minutes.",
        "citas": [{"chunk_id": "SLA-B#tiempos", "fragmento": "response within 30 minutes"}],
    }
    data.update(overrides)
    return data


def validate(data: dict, chunks: dict | None = CHUNKS) -> RespuestaConCitas:
    context = {"chunks": chunks} if chunks is not None else None
    return RespuestaConCitas.model_validate(data, context=context)


def test_valid_citation():
    assert validate(answer()).citas[0].chunk_id == "SLA-B#tiempos"


def test_citation_outside_context_is_rejected():
    data = answer(citas=[{"chunk_id": "RB-01#sintomas", "fragmento": "response within 30 minutes"}])
    with pytest.raises(ValidationError, match="not in the retrieved context"):
        validate(data)


def test_fragment_must_appear_in_chunk():
    data = answer(citas=[{"chunk_id": "SLA-B#tiempos", "fragmento": "response within 10 minutes"}])
    with pytest.raises(ValidationError, match="does not appear verbatim"):
        validate(data)


def test_fragment_match_ignores_case_spaces_and_quotes():
    fragment = '"Total   OUTAGE: response within 30 minutes."'
    data = answer(citas=[{"chunk_id": "SLA-B#tiempos", "fragmento": fragment}])
    assert validate(data)


def test_found_requires_citations():
    with pytest.raises(ValidationError, match="at least one citation"):
        validate(answer(citas=[]))


def test_not_found_allows_empty_citations():
    data = answer(encontrado=False, respuesta="No information found.", citas=[])
    assert validate(data).encontrado is False


def test_without_context_only_shape_is_checked():
    data = answer(citas=[{"chunk_id": "X#y", "fragmento": "anything"}])
    assert validate(data, chunks=None).citas[0].chunk_id == "X#y"


def test_normalize():
    assert normalize("  «Hello   World…»  ") == "hello world"
