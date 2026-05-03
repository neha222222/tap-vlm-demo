"""Tests for parse_vlm_output."""

import pytest

from vlm_grader.parser import ParseError, parse_vlm_output


def _valid_json():
    return (
        '{"creativity": 3, "critical_thinking": 4, "problem_solving": 2, '
        '"agency": 3, "justification": "Strong visual narrative.", '
        '"confidence": 0.7}'
    )


def test_parse_clean_json():
    s = parse_vlm_output(_valid_json())
    assert s.creativity == 3
    assert s.confidence == 0.7


def test_parse_with_leading_prose():
    raw = "Sure, here is the result:\n\n" + _valid_json() + "\n\nLet me know."
    s = parse_vlm_output(raw)
    assert s.critical_thinking == 4


def test_parse_with_code_fence():
    raw = "```json\n" + _valid_json() + "\n```"
    s = parse_vlm_output(raw)
    assert s.problem_solving == 2


def test_parse_trailing_comma_recoverable():
    raw = (
        '{"creativity": 1, "critical_thinking": 2, "problem_solving": 3, '
        '"agency": 4, "justification": "trailing comma",}'
    )
    s = parse_vlm_output(raw)
    assert s.agency == 4


def test_parse_no_json_raises():
    with pytest.raises(ParseError):
        parse_vlm_output("plain prose with no JSON")


def test_parse_unbalanced_json_raises():
    with pytest.raises(ParseError):
        parse_vlm_output("{ this is not balanced")


def test_parse_missing_required_field_raises():
    raw = '{"creativity": 1, "critical_thinking": 1, "problem_solving": 1, "justification": "missing agency"}'
    with pytest.raises(ParseError):
        parse_vlm_output(raw)


def test_parse_score_out_of_range_raises():
    raw = (
        '{"creativity": 7, "critical_thinking": 2, "problem_solving": 2, '
        '"agency": 2, "justification": "ok"}'
    )
    with pytest.raises(ParseError):
        parse_vlm_output(raw)


def test_parse_non_string_input_raises():
    with pytest.raises(ParseError):
        parse_vlm_output(123)  # type: ignore


def test_parse_strict_off_allows_missing_confidence():
    raw = (
        '{"creativity": 1, "critical_thinking": 2, "problem_solving": 3, '
        '"agency": 4, "justification": "ok"}'
    )
    s = parse_vlm_output(raw, strict=False)
    assert s.confidence is None
