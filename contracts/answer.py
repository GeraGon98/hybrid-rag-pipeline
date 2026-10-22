"""Assistant response contract: answer text with verifiable citations.

With `context={"chunks": {chunk_id: text}}`, every citation must reference a chunk in
the supplied context and its excerpt must appear verbatim. Without context, validation
only checks the shape and the rule `encontrado=True` => at least one citation.
"""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, Field, ValidationInfo, model_validator

_EDGE_CHARS = " \t\n\"'“”‘’«».…"


class Cita(BaseModel):
    chunk_id: str = Field(min_length=1, description="Exact identifier of the cited chunk.")
    fragmento: str = Field(
        min_length=3,
        max_length=400,
        description="Verbatim excerpt from the chunk supporting the answer.",
    )


class RespuestaConCitas(BaseModel):
    encontrado: bool = Field(
        description="false when the documents do not contain the answer."
    )
    respuesta: str = Field(min_length=1, description="Concise answer in English.")
    citas: list[Cita] = Field(description="Citations supporting the answer.")

    @model_validator(mode="after")
    def _check_citations(self, info: ValidationInfo) -> RespuestaConCitas:
        if self.encontrado and not self.citas:
            raise ValueError("encontrado=true requires at least one citation.")
        chunks: dict[str, str] | None = (info.context or {}).get("chunks")
        if chunks is None:
            return self
        for index, cita in enumerate(self.citas):
            text = chunks.get(cita.chunk_id)
            if text is None:
                allowed = ", ".join(sorted(chunks))
                raise ValueError(
                    f"citas[{index}].chunk_id {cita.chunk_id!r} is not in the retrieved "
                    f"context. Use only: {allowed}."
                )
            if normalize(cita.fragmento) not in normalize(text):
                raise ValueError(
                    f"citas[{index}].fragmento does not appear verbatim in {cita.chunk_id}. "
                    "Copy the exact sentence from the document."
                )
        return self


def normalize(text: str) -> str:
    """Lowercase text, collapse whitespace, and trim surrounding quotes or periods."""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip(_EDGE_CHARS)
