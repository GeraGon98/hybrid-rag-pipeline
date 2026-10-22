"""Structured output: request JSON, validate it with Pydantic, and repair failures."""

from __future__ import annotations

import copy
import json
import re
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from core.llm_client import LLMClient

T = TypeVar("T", bound=BaseModel)

# Keywords removed from the schema copy sent to the provider:
# - maxLength, minLength, and pattern: not every provider accepts them in strict mode;
#   Pydantic enforces them during local validation.
# - title and default: noise for the model; `default` also conflicts with strict mode.
# Local validation always uses the complete model, so no rule is lost.
UNSUPPORTED_KEYWORDS = frozenset({"maxLength", "minLength", "pattern", "title", "default"})

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class StructuredOutputError(RuntimeError):
    """The model did not produce valid output after all repair attempts."""

    def __init__(self, schema_name: str, attempts: int, last_error: str) -> None:
        super().__init__(
            f"No valid {schema_name} was produced after {attempts} attempt(s). "
            f"Last error: {last_error}"
        )
        self.attempts = attempts
        self.last_error = last_error


def sanitize_schema(schema: dict, *, strict: bool = True) -> dict:
    """Return a provider-compatible copy of the JSON Schema.

    Remove `UNSUPPORTED_KEYWORDS` and inline `$ref`/`$defs`. With `strict=True`, also
    set `additionalProperties: false` and mark every property as `required`.
    """
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def clean(node):
        if isinstance(node, list):
            return [clean(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            return clean(copy.deepcopy(defs[name]))
        result = {k: clean(v) for k, v in node.items() if k not in UNSUPPORTED_KEYWORDS}
        if "properties" in node:
            # `properties` maps field names to schemas; its keys are not schema keywords.
            result["properties"] = {k: clean(v) for k, v in node["properties"].items()}
            if strict:
                result["required"] = list(node["properties"])
                result["additionalProperties"] = False
        return result

    return clean(schema)


def schema_instruction(schema: dict) -> str:
    return (
        "Respond only with a valid JSON object matching this JSON Schema. Do not include "
        "additional text or code fences:\n" + json.dumps(schema, ensure_ascii=False)
    )


def repair_message(error: ValidationError) -> str:
    """Describe validation errors in a message the model can correct."""
    lines = []
    for err in error.errors():
        where = ".".join(str(part) for part in err["loc"]) or "response"
        message = err["msg"].removeprefix("Value error, ")
        lines.append(f"- {where}: {message}")
    return (
        "Your previous response does not match the required format. Errors:\n"
        + "\n".join(lines)
        + "\nRespond again with only the corrected JSON."
    )


def _with_instruction(messages: list[dict], instruction: str) -> list[dict]:
    messages = [dict(m) for m in messages]
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = f"{messages[0]['content']}\n\n{instruction}"
    else:
        messages.insert(0, {"role": "system", "content": instruction})
    return messages


def _strip_fences(text: str) -> str:
    text = text.strip()
    match = _FENCE.match(text)
    return match.group(1) if match else text


def generate_structured(
    client: LLMClient,
    messages: list[dict],
    schema: type[T],
    max_repairs: int = 2,
    *,
    log_extra: dict | None = None,
    validation_context: dict | None = None,
    **params,
) -> T:
    """Request `schema` from the model and return validated output, repairing failures.

    The strategy depends on the first provider in the chain:
    1. Strict `json_schema` when `supports_json_schema`.
    2. `json_object` plus the schema in the system message when `supports_json_object`.
    3. The schema in the system message only.
    Validators receive `validation_context` as `info.context` (for example, retrieved
    chunks used to validate citations). Provider errors propagate without repair.
    """
    provider = client.providers[0]
    sent_schema = sanitize_schema(schema.model_json_schema())
    history = [dict(m) for m in messages]

    if getattr(provider, "supports_json_schema", False):
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": sent_schema, "strict": True},
        }
    elif getattr(provider, "supports_json_object", False):
        response_format = {"type": "json_object"}
        history = _with_instruction(history, schema_instruction(sent_schema))
    else:
        response_format = None
        history = _with_instruction(history, schema_instruction(sent_schema))

    last_error = ""
    for attempt in range(max_repairs + 1):
        extra = {**(log_extra or {}), "structured_schema": schema.__name__}
        extra["repair_attempt"] = attempt
        response = client.generate(
            history, log_extra=extra, response_format=response_format, **params
        )
        try:
            return schema.model_validate_json(
                _strip_fences(response.text), context=validation_context
            )
        except ValidationError as e:
            last_error = repair_message(e)
            history.append({"role": "assistant", "content": response.text})
            history.append({"role": "user", "content": last_error})

    raise StructuredOutputError(schema.__name__, max_repairs + 1, last_error)
