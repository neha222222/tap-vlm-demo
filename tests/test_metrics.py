"""Tests for QWK + per-dimension MAE + format validity."""

import pytest

from vlm_grader.metrics import (
    format_validity_rate,
    per_dimension_mae,
    quadratic_weighted_kappa,
    subset_accuracy,
)
from vlm_grader.rubric import RubricScore


# --- QWK ------------------------------------------------------------------


def test_qwk_perfect_agreement():
    a = [1, 2, 3, 4, 1, 2, 3, 4]
    b = [1, 2, 3, 4, 1, 2, 3, 4]
    assert quadratic_weighted_kappa(a, b) == pytest.approx(1.0)


def test_qwk_perfect_disagreement_low():
    a = [1] * 4 + [4] * 4
    b = [4] * 4 + [1] * 4
    assert quadratic_weighted_kappa(a, b) < -0.5


def test_qwk_partial_agreement_positive():
    a = [1, 2, 3, 4, 1, 2, 3, 4]
    b = [1, 2, 3, 4, 2, 2, 3, 4]
    qwk = quadratic_weighted_kappa(a, b)
    assert 0.5 < qwk < 1.0


def test_qwk_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        quadratic_weighted_kappa([1, 2], [1])


def test_qwk_empty_inputs_rejected():
    with pytest.raises(ValueError):
        quadratic_weighted_kappa([], [])


def test_qwk_score_out_of_range_raises():
    with pytest.raises(ValueError):
        quadratic_weighted_kappa([5, 1], [1, 1])


# --- MAE ------------------------------------------------------------------


def _score(c=1, ct=1, ps=1, ag=1, j="ok"):
    return RubricScore(c, ct, ps, ag, justification=j)


def test_mae_zero_for_identical():
    pred = [_score(2, 2, 2, 2)] * 3
    actual = [_score(2, 2, 2, 2)] * 3
    mae = per_dimension_mae(pred, actual)
    assert all(v == 0.0 for v in mae.values())


def test_mae_per_dim():
    pred = [_score(1, 2, 3, 4)]
    actual = [_score(2, 2, 3, 4)]
    mae = per_dimension_mae(pred, actual)
    assert mae["creativity"] == 1.0
    assert mae["critical_thinking"] == 0.0


def test_mae_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        per_dimension_mae([_score()], [])


# --- Subset accuracy ------------------------------------------------------


def test_subset_accuracy_full_match():
    pred = [_score(1, 1, 1, 1)] * 3
    actual = [_score(1, 1, 1, 1)] * 3
    assert subset_accuracy(pred, actual) == 1.0


def test_subset_accuracy_no_match():
    pred = [_score(1, 1, 1, 1)]
    actual = [_score(2, 1, 1, 1)]
    assert subset_accuracy(pred, actual) == 0.0


# --- Format validity ------------------------------------------------------


def test_format_validity_rate_all_valid():
    raws = [
        '{"creativity":1,"critical_thinking":1,"problem_solving":1,"agency":1,"justification":"ok"}',
    ] * 5
    rate, parsed = format_validity_rate(raws)
    assert rate == 1.0
    assert all(p is not None for p in parsed)


def test_format_validity_rate_mixed():
    raws = [
        '{"creativity":1,"critical_thinking":1,"problem_solving":1,"agency":1,"justification":"ok"}',
        "not json",
        '{"creativity":2,"critical_thinking":2,"problem_solving":2,"agency":2,"justification":"ok"}',
    ]
    rate, parsed = format_validity_rate(raws)
    assert rate == pytest.approx(2 / 3)
    assert parsed[1] is None


def test_format_validity_rate_empty_rejected():
    with pytest.raises(ValueError):
        format_validity_rate([])
