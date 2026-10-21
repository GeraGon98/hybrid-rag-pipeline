"""Contratos de salida: `RespuestaConCitas` y la generación estructurada con reparación."""

from contracts.answer import Cita, RespuestaConCitas
from contracts.structured_output import StructuredOutputError, generate_structured

__all__ = ["Cita", "RespuestaConCitas", "StructuredOutputError", "generate_structured"]
