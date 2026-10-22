"""Output contracts: `RespuestaConCitas` and repairable structured generation."""

from contracts.answer import Cita, RespuestaConCitas
from contracts.structured_output import StructuredOutputError, generate_structured

__all__ = ["Cita", "RespuestaConCitas", "StructuredOutputError", "generate_structured"]
