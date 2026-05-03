"""JSON schema for VLM rubric outputs + lightweight validator.

A self-contained validator (no jsonschema dep) so the package stays
light. The schema is exposed for use with constrained-decoding
libraries (outlines, xgrammar) when those are wired in upstream.
"""

from __future__ import annotations

from typing import Any, Mapping

from vlm_grader.rubric import MAX_JUSTIFICATION_LEN, RUBRIC_DIMENSIONS


JSON_SCHEMA: dict = {
    "type": "object",
    "required": [*RUBRIC_DIMENSIONS, "justification"],
    "additionalProperties": False,
    "properties": {
        **{
            dim: {"type": "integer", "minimum": 1, "maximum": 4}
            for dim in RUBRIC_DIMENSIONS
        },
        "justification": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_JUSTIFICATION_LEN,
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


class SchemaValidationError(ValueError):
    """Raised when a payload does not match :data:`JSON_SCHEMA`."""


def validate_against_schema(payload: Mapping[str, Any]) -> None:
    """Validate ``payload`` against :data:`JSON_SCHEMA`.

    Raises :class:`SchemaValidationError` on mismatch.
    """
    if not isinstance(payload, Mapping):
        raise SchemaValidationError(f"expected object, got {type(payload).__name__}")

    required = JSON_SCHEMA["required"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise SchemaValidationError(f"missing required fields: {missing}")

    allowed = set(JSON_SCHEMA["properties"].keys())
    extra = [k for k in payload if k not in allowed]
    if extra:
        raise SchemaValidationError(f"unexpected fields: {extra}")

    for dim in RUBRIC_DIMENSIONS:
        v = payload[dim]
        if not isinstance(v, int) or isinstance(v, bool):
            raise SchemaValidationError(f"{dim} must be int, got {type(v).__name__}")
        if not 1 <= v <= 4:
            raise SchemaValidationError(f"{dim} must be in [1, 4], got {v}")

    j = payload["justification"]
    if not isinstance(j, str):
        raise SchemaValidationError("justification must be string")
    if not 1 <= len(j) <= MAX_JUSTIFICATION_LEN:
        raise SchemaValidationError(
            f"justification length {len(j)} outside [1, {MAX_JUSTIFICATION_LEN}]"
        )

    if "confidence" in payload:
        c = payload["confidence"]
        if not isinstance(c, (int, float)) or isinstance(c, bool):
            raise SchemaValidationError("confidence must be number")
        if not 0.0 <= float(c) <= 1.0:
            raise SchemaValidationError(f"confidence must be in [0, 1], got {c}")
