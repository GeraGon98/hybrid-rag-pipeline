from __future__ import annotations

import pytest
from pydantic import ValidationError

from contracts.answer import RespuestaConCitas, normalize

CHUNKS = {
    "SLA-B#tiempos": "Falla total: respuesta en 30 minutos y resolución en 4 horas.",
    "SLA-B#disponibilidad": "Disponibilidad mensual comprometida de 99.7 % por circuito.",
}


def answer(**overrides) -> dict:
    data = {
        "encontrado": True,
        "respuesta": "NetLink responde en 30 minutos.",
        "citas": [{"chunk_id": "SLA-B#tiempos", "fragmento": "respuesta en 30 minutos"}],
    }
    data.update(overrides)
    return data


def validate(data: dict, chunks: dict | None = CHUNKS) -> RespuestaConCitas:
    context = {"chunks": chunks} if chunks is not None else None
    return RespuestaConCitas.model_validate(data, context=context)


def test_valid_citation():
    assert validate(answer()).citas[0].chunk_id == "SLA-B#tiempos"


def test_citation_outside_context_is_rejected():
    data = answer(citas=[{"chunk_id": "RB-01#sintomas", "fragmento": "respuesta en 30 minutos"}])
    with pytest.raises(ValidationError, match="no está en el contexto"):
        validate(data)


def test_fragment_must_appear_in_chunk():
    data = answer(citas=[{"chunk_id": "SLA-B#tiempos", "fragmento": "respuesta en 10 minutos"}])
    with pytest.raises(ValidationError, match="no aparece textual"):
        validate(data)


def test_fragment_match_ignores_case_spaces_and_quotes():
    fragment = '"Falla   TOTAL: respuesta en 30 minutos."'
    data = answer(citas=[{"chunk_id": "SLA-B#tiempos", "fragmento": fragment}])
    assert validate(data)


def test_found_requires_citations():
    with pytest.raises(ValidationError, match="al menos una cita"):
        validate(answer(citas=[]))


def test_not_found_allows_empty_citations():
    data = answer(encontrado=False, respuesta="No encontré información.", citas=[])
    assert validate(data).encontrado is False


def test_without_context_only_shape_is_checked():
    data = answer(citas=[{"chunk_id": "X#y", "fragmento": "cualquier cosa"}])
    assert validate(data, chunks=None).citas[0].chunk_id == "X#y"


def test_normalize():
    assert normalize("  «Hola   Mundo…»  ") == "hola mundo"
