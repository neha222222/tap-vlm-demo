"""Tests for the evaluate_grader harness."""


import pytest

from vlm_grader.eval import EvalReport, evaluate_grader
from vlm_grader.inference import MockVLM
from vlm_grader.rubric import RubricScore
from vlm_grader.synthetic import (
    make_synthetic_dataset,
    synthetic_artifacts_for_eval,
)


def _gold_score(c=2, ct=2, ps=2, ag=2):
    return RubricScore(c, ct, ps, ag, justification="gold")


def test_perfect_grader_high_qwk():
    """If MockVLM returns the gold scores directly, QWK should be 1."""
    artifacts = synthetic_artifacts_for_eval(n=20, seed=0)

    # Build mock outputs that match the gold exactly.
    mock_scores = []
    for _, _, gold in artifacts:
        mock_scores.append(
            {
                "creativity": gold.creativity,
                "critical_thinking": gold.critical_thinking,
                "problem_solving": gold.problem_solving,
                "agency": gold.agency,
                "justification": gold.justification,
                "confidence": 0.9,
            }
        )
    grader = MockVLM(scores=mock_scores, latency_seconds=0.0)
    report = evaluate_grader(grader, artifacts)
    assert isinstance(report, EvalReport)
    assert report.format_validity_rate == 1.0
    assert report.subset_accuracy == 1.0
    for dim in report.qwk_per_dimension:
        assert report.qwk_per_dimension[dim] == pytest.approx(1.0)
    for dim in report.mae_per_dimension:
        assert report.mae_per_dimension[dim] == 0.0


def test_drifted_grader_lower_qwk():
    """A grader biased by +1 should still have positive QWK but < 1."""
    ds = make_synthetic_dataset(n=20, seed=0)
    artifacts = [(a.image_bytes, "p", a.gold) for a in ds.artifacts]
    drifted = []
    for a in ds.artifacts:
        drifted.append(
            {
                "creativity": min(4, a.gold.creativity + 1),
                "critical_thinking": min(4, a.gold.critical_thinking + 1),
                "problem_solving": min(4, a.gold.problem_solving + 1),
                "agency": min(4, a.gold.agency + 1),
                "justification": "drifted",
                "confidence": 0.5,
            }
        )
    grader = MockVLM(scores=drifted, latency_seconds=0.0)
    report = evaluate_grader(grader, artifacts)
    assert report.subset_accuracy < 1.0
    # QWK should still be moderate-high since drift is uniform.
    for k in report.qwk_per_dimension.values():
        assert 0.0 <= k <= 1.0


def test_partial_format_failures():
    artifacts = synthetic_artifacts_for_eval(n=10, seed=0)
    grader = MockVLM(latency_seconds=0.0, malformed_rate=0.5, seed=0)
    report = evaluate_grader(grader, artifacts)
    assert 0.0 <= report.format_validity_rate <= 1.0
    assert len(report.failures) > 0


def test_empty_artifacts_rejected():
    grader = MockVLM()
    with pytest.raises(ValueError):
        evaluate_grader(grader, [])


def test_summary_string_contains_metrics():
    artifacts = synthetic_artifacts_for_eval(n=5, seed=0)
    grader = MockVLM(latency_seconds=0.0)
    report = evaluate_grader(grader, artifacts)
    s = report.summary()
    assert "format_validity" in s
    assert "QWK per dimension" in s
    assert "projected cost" in s.lower() or "₹" in s
