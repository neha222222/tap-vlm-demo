"""Tests for the rubric parsing + QWK metric."""

import pytest

from rubric import (
    JSON_SCHEMA,
    RUBRIC_DIMENSIONS,
    RubricScore,
    build_prompt,
    parse_vlm_output,
    quadratic_weighted_kappa,
)


# --- RubricScore validation -----------------------------------------------


def test_valid_score():
    s = RubricScore(
        creativity=3,
        critical_thinking=2,
        problem_solving=4,
        agency=3,
        justification="The drawing shows multiple unique elements.",
        confidence=0.85,
    )
    assert s.creativity == 3
    d = s.to_dict()
    assert set(d.keys()) == {*RUBRIC_DIMENSIONS, "justification", "confidence"}


def test_score_out_of_range_rejected():
    with pytest.raises(ValueError):
        RubricScore(0, 1, 1, 1, justification="x")
    with pytest.raises(ValueError):
        RubricScore(1, 1, 1, 5, justification="x")


def test_empty_justification_rejected():
    with pytest.raises(ValueError):
        RubricScore(1, 1, 1, 1, justification="   ")


def test_too_long_justification_rejected():
    with pytest.raises(ValueError):
        RubricScore(1, 1, 1, 1, justification="x" * 281)


def test_invalid_confidence_rejected():
    with pytest.raises(ValueError):
        RubricScore(1, 1, 1, 1, justification="x", confidence=1.5)


# --- Output parsing -------------------------------------------------------


def test_parse_clean_json():
    raw = (
        '{"creativity": 3, "critical_thinking": 4, "problem_solving": 2, '
        '"agency": 3, "justification": "Strong visual narrative.", '
        '"confidence": 0.7}'
    )
    s = parse_vlm_output(raw)
    assert s.creativity == 3
    assert s.confidence == 0.7


def test_parse_json_with_leading_prose():
    raw = (
        "Sure, here is the rubric scoring:\n\n"
        '{"creativity": 2, "critical_thinking": 3, "problem_solving": 2, '
        '"agency": 1, "justification": "Sketch is simple but on-task."}\n\n'
        "Let me know if you need adjustments."
    )
    s = parse_vlm_output(raw)
    assert s.creativity == 2
    assert s.agency == 1
    assert s.confidence is None


def test_parse_no_json_raises():
    with pytest.raises(ValueError):
        parse_vlm_output("This is just prose with no JSON.")


def test_parse_invalid_json_raises():
    with pytest.raises(ValueError):
        parse_vlm_output("{this is not valid json}")


# --- Prompt + schema ------------------------------------------------------


def test_build_prompt_mentions_all_dimensions():
    prompt = build_prompt()
    for dim in RUBRIC_DIMENSIONS:
        assert dim in prompt


def test_schema_has_all_required_fields():
    required = set(JSON_SCHEMA["required"])
    assert {"creativity", "critical_thinking", "problem_solving", "agency", "justification"} <= required


# --- QWK metric -----------------------------------------------------------


def test_qwk_perfect_agreement():
    a = [1, 2, 3, 4, 1, 2, 3, 4]
    b = [1, 2, 3, 4, 1, 2, 3, 4]
    assert quadratic_weighted_kappa(a, b) == pytest.approx(1.0)


def test_qwk_perfect_disagreement_low():
    # All raters confidently disagree by maximum distance.
    a = [1] * 4 + [4] * 4
    b = [4] * 4 + [1] * 4
    qwk = quadratic_weighted_kappa(a, b)
    assert qwk < -0.5


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
