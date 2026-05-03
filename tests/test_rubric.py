"""Tests for RubricScore validation."""

import pytest

from vlm_grader.rubric import RUBRIC_DIMENSIONS, RubricScore


def test_valid_score():
    s = RubricScore(3, 2, 4, 3, justification="ok", confidence=0.9)
    assert s.creativity == 3
    d = s.to_dict()
    assert set(d.keys()) == {*RUBRIC_DIMENSIONS, "justification", "confidence"}


def test_overall_is_mean():
    s = RubricScore(2, 2, 2, 2, justification="ok")
    assert s.overall() == 2.0
    s2 = RubricScore(1, 2, 3, 4, justification="ok")
    assert s2.overall() == 2.5


@pytest.mark.parametrize("dim", RUBRIC_DIMENSIONS)
def test_dim_out_of_range_rejected(dim):
    kwargs = {d: 1 for d in RUBRIC_DIMENSIONS}
    kwargs["justification"] = "ok"
    kwargs[dim] = 5
    with pytest.raises(ValueError):
        RubricScore(**kwargs)


def test_empty_justification_rejected():
    with pytest.raises(ValueError):
        RubricScore(1, 1, 1, 1, justification="   ")


def test_too_long_justification_rejected():
    with pytest.raises(ValueError):
        RubricScore(1, 1, 1, 1, justification="x" * 281)


def test_invalid_confidence_rejected():
    with pytest.raises(ValueError):
        RubricScore(1, 1, 1, 1, justification="ok", confidence=1.5)


def test_non_int_dim_rejected():
    with pytest.raises(ValueError):
        RubricScore(creativity="3", critical_thinking=1, problem_solving=1, agency=1, justification="x")  # type: ignore
