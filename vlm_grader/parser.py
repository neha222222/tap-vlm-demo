"""Parse VLM string outputs into validated :class:`RubricScore` objects.

Tolerant of leading/trailing prose, common JSON quirks (single quotes,
trailing commas), and known model-specific failure modes.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from vlm_grader.rubric import RubricScore
from vlm_grader.schema import SchemaValidationError, validate_against_schema


_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


class ParseError(ValueError):
    """Raised when a VLM output cannot be parsed into a RubricScore."""


def _extract_json_blob(text: str) -> str:
    """Find the first balanced ``{...}`` substring."""
    start = text.find("{")
    if start == -1:
        raise ParseError("no '{' found in VLM output")
    # Walk to the matching brace, ignoring braces inside strings.
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ParseError("unbalanced JSON in VLM output")


def _normalize_json(blob: str) -> str:
    """Apply small fixups for common LLM JSON quirks."""
    # Strip code-fence markers if present.
    blob = blob.strip()
    # Remove trailing commas before } or ].
    blob = _TRAILING_COMMA_RE.sub(r"\1", blob)
    return blob


def parse_vlm_output(text: str, *, strict: bool = True) -> RubricScore:
    """Parse a raw VLM string response into a :class:`RubricScore`.

    Parameters
    ----------
    text:
        The raw VLM output.
    strict:
        When ``True`` (default), the payload must validate against
        :data:`JSON_SCHEMA`. When ``False``, missing optional fields
        like ``confidence`` are filled with ``None``.
    """
    if not isinstance(text, str):
        raise ParseError(f"expected str, got {type(text).__name__}")

    blob = _extract_json_blob(text)
    blob = _normalize_json(blob)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON: {exc}") from exc

    if strict:
        try:
            validate_against_schema(data)
        except SchemaValidationError as exc:
            raise ParseError(f"schema validation failed: {exc}") from exc

    confidence: Optional[float]
    if "confidence" in data and data["confidence"] is not None:
        confidence = float(data["confidence"])
    else:
        confidence = None

    return RubricScore(
        creativity=int(data["creativity"]),
        critical_thinking=int(data["critical_thinking"]),
        problem_solving=int(data["problem_solving"]),
        agency=int(data["agency"]),
        justification=str(data["justification"]),
        confidence=confidence,
    )
