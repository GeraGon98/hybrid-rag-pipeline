"""Registro estructurado de llamadas al LLM en formato JSONL (una línea por intento).

Regla de oro: aquí nunca se escriben llaves, headers, ni el texto del prompt, de la
respuesta o de los documentos. Del prompt solo se guarda su tamaño en caracteres
(`prompt_chars`) y, de la recuperación, la estrategia y los `chunk_ids` usados.

Campos opcionales: `prompt_name`, `prompt_version`, `structured_schema`, `repair_attempt`,
`retrieval_strategy` y `chunk_ids`. Solo se escriben cuando vienen.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path


class CallLogger:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def log(
        self,
        *,
        provider: str,
        model: str | None,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        cost_usd: float,
        fallback: bool,
        success: bool,
        error_type: str | None,
        prompt_chars: int,
        prompt_name: str | None = None,
        prompt_version: int | None = None,
        structured_schema: str | None = None,
        repair_attempt: int | None = None,
        retrieval_strategy: str | None = None,
        chunk_ids: list[str] | None = None,
    ) -> dict:
        """Agrega un evento al archivo y lo devuelve.

        Cualquier campo opcional que no esté en la firma lanza `TypeError`: así nadie
        puede colar texto del prompt al log por accidente.
        """
        event = {
            "request_id": uuid.uuid4().hex[:8],
            "timestamp": datetime.now(UTC).isoformat(),
            "provider": provider,
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": round(latency_ms, 1),
            "cost_usd": cost_usd,
            "fallback": fallback,
            "success": success,
            "error_type": error_type,
            "prompt_chars": prompt_chars,
        }
        optional = {
            "prompt_name": prompt_name,
            "prompt_version": prompt_version,
            "structured_schema": structured_schema,
            "repair_attempt": repair_attempt,
            "retrieval_strategy": retrieval_strategy,
            "chunk_ids": chunk_ids,
        }
        event.update({key: value for key, value in optional.items() if value is not None})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event
