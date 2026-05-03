"""Tests for MockVLM + grade trace shape."""

import pytest

from vlm_grader.inference import MockVLM
from vlm_grader.parser import parse_vlm_output


def test_mock_returns_parseable_output():
    mock = MockVLM(latency_seconds=0.0)
    trace = mock.grade(image=None, prompt="test")
    score = parse_vlm_output(trace.raw_output)
    assert 1 <= score.creativity <= 4


def test_mock_iterates_through_scores():
    scores = [
        {
            "creativity": 1, "critical_thinking": 1, "problem_solving": 1,
            "agency": 1, "justification": "first", "confidence": 0.5,
        },
        {
            "creativity": 4, "critical_thinking": 4, "problem_solving": 4,
            "agency": 4, "justification": "second", "confidence": 0.9,
        },
    ]
    mock = MockVLM(scores=scores, latency_seconds=0.0)
    s1 = parse_vlm_output(mock.grade(None, "x").raw_output)
    s2 = parse_vlm_output(mock.grade(None, "x").raw_output)
    assert s1.creativity == 1
    assert s2.creativity == 4


def test_mock_with_prose_wrapping_still_parses():
    mock = MockVLM(wrap_in_prose=True, latency_seconds=0.0)
    trace = mock.grade(image=None, prompt="prompt")
    parse_vlm_output(trace.raw_output)


def test_mock_malformed_rate_produces_failures():
    mock = MockVLM(latency_seconds=0.0, malformed_rate=1.0, seed=0)
    trace = mock.grade(image=None, prompt="x")
    # Malformed output should fail strict parsing.
    from vlm_grader.parser import ParseError
    with pytest.raises(ParseError):
        parse_vlm_output(trace.raw_output)


def test_mock_records_token_counts():
    mock = MockVLM(latency_seconds=0.0)
    trace = mock.grade(image=None, prompt="hello world")
    assert trace.prompt_tokens is not None
    assert trace.completion_tokens is not None
    assert trace.prompt_tokens > 0


def test_invalid_malformed_rate_rejected():
    with pytest.raises(ValueError):
        MockVLM(malformed_rate=2.0)
