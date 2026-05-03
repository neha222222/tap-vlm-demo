"""Tests for schema validation."""

import pytest

from vlm_grader.schema import JSON_SCHEMA, SchemaValidationError, validate_against_schema


def _valid():
    return {
        "creativity": 1,
        "critical_thinking": 2,
        "problem_solving": 3,
        "agency": 4,
        "justification": "ok",
        "confidence": 0.5,
    }


def test_schema_required_fields():
    required = set(JSON_SCHEMA["required"])
    assert {"creativity", "critical_thinking", "problem_solving", "agency", "justification"} <= required


def test_valid_payload_passes():
    validate_against_schema(_valid())


def test_missing_required_raises():
    bad = _valid()
    del bad["agency"]
    with pytest.raises(SchemaValidationError):
        validate_against_schema(bad)


def test_extra_field_raises():
    bad = _valid()
    bad["extra_field"] = "not allowed"
    with pytest.raises(SchemaValidationError):
        validate_against_schema(bad)


def test_score_out_of_range_raises():
    bad = _valid()
    bad["creativity"] = 5
    with pytest.raises(SchemaValidationError):
        validate_against_schema(bad)


def test_score_wrong_type_raises():
    bad = _valid()
    bad["creativity"] = "3"  # type: ignore
    with pytest.raises(SchemaValidationError):
        validate_against_schema(bad)


def test_bool_score_rejected():
    bad = _valid()
    bad["creativity"] = True  # type: ignore
    with pytest.raises(SchemaValidationError):
        validate_against_schema(bad)


def test_too_long_justification_raises():
    bad = _valid()
    bad["justification"] = "x" * 500
    with pytest.raises(SchemaValidationError):
        validate_against_schema(bad)


def test_invalid_confidence_raises():
    bad = _valid()
    bad["confidence"] = 2.0
    with pytest.raises(SchemaValidationError):
        validate_against_schema(bad)


def test_confidence_optional():
    payload = _valid()
    del payload["confidence"]
    validate_against_schema(payload)


def test_non_object_raises():
    with pytest.raises(SchemaValidationError):
        validate_against_schema([1, 2, 3])  # type: ignore
