"""Contrato de la respuesta del asistente: texto más citas verificables.

Con `context={"chunks": {chunk_id: texto}}` el validador exige que cada cita apunte a un
chunk que estuvo en el contexto y que su fragmento aparezca textual en ese chunk. Sin
contexto solo se validan la forma y la regla `encontrado=True` => al menos una cita.
"""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, Field, ValidationInfo, model_validator

_EDGE_CHARS = " \t\n\"'“”‘’«».…"


class Cita(BaseModel):
    chunk_id: str = Field(min_length=1, description="Identificador exacto del chunk citado.")
    fragmento: str = Field(
        min_length=3,
        max_length=400,
        description="Frase copiada textualmente del chunk que respalda la respuesta.",
    )


class RespuestaConCitas(BaseModel):
    encontrado: bool = Field(
        description="false si los documentos no contienen la respuesta a la pregunta."
    )
    respuesta: str = Field(min_length=1, description="Respuesta breve en español.")
    citas: list[Cita] = Field(description="Citas que respaldan la respuesta.")

    @model_validator(mode="after")
    def _check_citations(self, info: ValidationInfo) -> RespuestaConCitas:
        if self.encontrado and not self.citas:
            raise ValueError("encontrado=true exige al menos una cita.")
        chunks: dict[str, str] | None = (info.context or {}).get("chunks")
        if chunks is None:
            return self
        for index, cita in enumerate(self.citas):
            text = chunks.get(cita.chunk_id)
            if text is None:
                allowed = ", ".join(sorted(chunks))
                raise ValueError(
                    f"citas[{index}].chunk_id {cita.chunk_id!r} no está en el contexto "
                    f"recuperado. Usa solo: {allowed}."
                )
            if normalize(cita.fragmento) not in normalize(text):
                raise ValueError(
                    f"citas[{index}].fragmento no aparece textual en {cita.chunk_id}. "
                    "Copia la frase exacta del documento."
                )
        return self


def normalize(text: str) -> str:
    """Minúsculas, espacios colapsados y sin comillas o puntos en los extremos."""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip(_EDGE_CHARS)
