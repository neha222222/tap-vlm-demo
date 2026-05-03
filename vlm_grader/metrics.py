"""Metrics for evaluating VLM rubric outputs.

- :func:`quadratic_weighted_kappa`: ordinal inter-rater agreement.
- :func:`per_dimension_mae`: per-rubric-dimension mean absolute error.
- :func:`format_validity_rate`: fraction of outputs that parse.
- :func:`subset_accuracy`: exact-match share of dimension scores.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from vlm_grader.parser import ParseError, parse_vlm_output
from vlm_grader.rubric import RUBRIC_DIMENSIONS, RubricScore


def quadratic_weighted_kappa(
    rater_a: Sequence[int],
    rater_b: Sequence[int],
    min_score: int = 1,
    max_score: int = 4,
) -> float:
    """Quadratic Weighted Kappa for two ordinal raters.

    Standard inter-rater agreement metric for ordinal rubrics. Range
    [-1, 1]. Implemented from scratch to keep the package light (no
    sklearn dependency).
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("rater_a and rater_b must be the same length")
    if not rater_a:
        raise ValueError("inputs must be non-empty")

    n_categories = max_score - min_score + 1
    observed = [[0] * n_categories for _ in range(n_categories)]
    for a, b in zip(rater_a, rater_b):
        if not (min_score <= a <= max_score) or not (min_score <= b <= max_score):
            raise ValueError(f"score out of [{min_score}, {max_score}]: a={a}, b={b}")
        observed[a - min_score][b - min_score] += 1

    hist_a = [sum(row) for row in observed]
    hist_b = [
        sum(observed[i][j] for i in range(n_categories))
        for j in range(n_categories)
    ]
    n = sum(hist_a)

    weights = [
        [((i - j) ** 2) / ((n_categories - 1) ** 2) for j in range(n_categories)]
        for i in range(n_categories)
    ]
    expected = [
        [(hist_a[i] * hist_b[j]) / n for j in range(n_categories)]
        for i in range(n_categories)
    ]
    num = sum(
        weights[i][j] * observed[i][j]
        for i in range(n_categories)
        for j in range(n_categories)
    )
    den = sum(
        weights[i][j] * expected[i][j]
        for i in range(n_categories)
        for j in range(n_categories)
    )
    if den == 0:
        return 1.0 if num == 0 else 0.0
    return 1.0 - num / den


def per_dimension_mae(
    predicted: Sequence[RubricScore],
    actual: Sequence[RubricScore],
) -> dict:
    """Per-rubric-dimension mean absolute error."""
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must be the same length")
    if not predicted:
        raise ValueError("inputs must be non-empty")
    out = {}
    for dim in RUBRIC_DIMENSIONS:
        diffs = [
            abs(getattr(p, dim) - getattr(a, dim)) for p, a in zip(predicted, actual)
        ]
        out[dim] = sum(diffs) / len(diffs)
    return out


def subset_accuracy(
    predicted: Sequence[RubricScore],
    actual: Sequence[RubricScore],
) -> float:
    """Fraction of artifacts where every dimension matches exactly."""
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must be the same length")
    if not predicted:
        raise ValueError("inputs must be non-empty")
    matches = 0
    for p, a in zip(predicted, actual):
        if all(getattr(p, d) == getattr(a, d) for d in RUBRIC_DIMENSIONS):
            matches += 1
    return matches / len(predicted)


def format_validity_rate(raw_outputs: Iterable[str]) -> Tuple[float, List[Optional[RubricScore]]]:
    """Run the parser over every raw output, return validity rate + parsed list.

    Returns
    -------
    rate:
        Fraction of outputs that parse successfully.
    parsed:
        List the same length as ``raw_outputs``, with ``None`` for
        unparseable entries.
    """
    raw_list = list(raw_outputs)
    if not raw_list:
        raise ValueError("raw_outputs must be non-empty")
    parsed: List[Optional[RubricScore]] = []
    valid = 0
    for raw in raw_list:
        try:
            score = parse_vlm_output(raw)
            parsed.append(score)
            valid += 1
        except ParseError:
            parsed.append(None)
    return valid / len(raw_list), parsed
