"""Tests for the synthetic dataset generator."""

import pytest

from vlm_grader.parser import parse_vlm_output
from vlm_grader.synthetic import (
    make_mock_vlm_outputs,
    make_synthetic_dataset,
)


def test_dataset_size_and_seed():
    ds1 = make_synthetic_dataset(n=20, seed=42)
    ds2 = make_synthetic_dataset(n=20, seed=42)
    assert len(ds1) == 20
    # Same seed -> same data.
    for a, b in zip(ds1.artifacts, ds2.artifacts):
        assert a.gold.creativity == b.gold.creativity


def test_different_seeds_produce_different_data():
    ds1 = make_synthetic_dataset(n=20, seed=1)
    ds2 = make_synthetic_dataset(n=20, seed=2)
    diffs = sum(
        1
        for a, b in zip(ds1.artifacts, ds2.artifacts)
        if a.gold.creativity != b.gold.creativity
    )
    assert diffs > 0


def test_mock_outputs_parse():
    ds = make_synthetic_dataset(n=10, seed=0)
    outs = make_mock_vlm_outputs(ds)
    for raw in outs:
        parse_vlm_output(raw)


def test_drift_changes_outputs():
    ds = make_synthetic_dataset(n=10, seed=0)
    raw_zero = make_mock_vlm_outputs(ds, error_dimension_drift=0)
    raw_pos = make_mock_vlm_outputs(ds, error_dimension_drift=1)
    parsed_zero = [parse_vlm_output(r) for r in raw_zero]
    parsed_pos = [parse_vlm_output(r) for r in raw_pos]
    diffs = sum(1 for a, b in zip(parsed_zero, parsed_pos) if a.creativity != b.creativity)
    # Most should differ unless gold was already at the ceiling.
    assert diffs >= 5


def test_malformed_indices_produce_unparseable():
    ds = make_synthetic_dataset(n=5, seed=0)
    outs = make_mock_vlm_outputs(ds, malformed_indices=[2])
    from vlm_grader.parser import ParseError
    for i, raw in enumerate(outs):
        if i == 2:
            with pytest.raises(ParseError):
                parse_vlm_output(raw)
        else:
            parse_vlm_output(raw)


def test_clamp_drift():
    """Drift should clamp into [1, 4]."""
    ds = make_synthetic_dataset(n=5, seed=0)
    outs = make_mock_vlm_outputs(ds, error_dimension_drift=10)
    for raw in outs:
        score = parse_vlm_output(raw)
        for dim in ("creativity", "critical_thinking", "problem_solving", "agency"):
            assert 1 <= getattr(score, dim) <= 4
