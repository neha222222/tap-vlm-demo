"""Tests for cost projection."""

import pytest

from vlm_grader.cost import CostModel, headroom_vs_target, project_cost_per_assessment


def test_default_projection():
    cost = project_cost_per_assessment(mean_latency_seconds=0.1)
    # 0.1s * 1.2 overhead / 3600 * ₹50 = ~₹0.00167
    assert cost == pytest.approx(0.001667, rel=0.05)


def test_batching_amortizes_cost():
    single = project_cost_per_assessment(mean_latency_seconds=0.1, batch_size=1)
    batched = project_cost_per_assessment(mean_latency_seconds=0.1, batch_size=10)
    assert batched == pytest.approx(single / 10)


def test_cache_hit_rate_reduces_cost():
    no_cache = project_cost_per_assessment(0.1, cost_model=CostModel(cache_hit_rate=0.0))
    half_cache = project_cost_per_assessment(0.1, cost_model=CostModel(cache_hit_rate=0.5))
    full_cache = project_cost_per_assessment(0.1, cost_model=CostModel(cache_hit_rate=1.0))
    assert half_cache == pytest.approx(no_cache / 2)
    assert full_cache == pytest.approx(0.0)


def test_invalid_latency_rejected():
    with pytest.raises(ValueError):
        project_cost_per_assessment(0.0)


def test_invalid_batch_size_rejected():
    with pytest.raises(ValueError):
        project_cost_per_assessment(0.1, batch_size=0)


def test_invalid_cache_hit_rate_rejected():
    with pytest.raises(ValueError):
        project_cost_per_assessment(0.1, cost_model=CostModel(cache_hit_rate=2.0))


def test_headroom_factor():
    h = headroom_vs_target(projected_cost=0.05, target_inr=0.10)
    assert h == 2.0


def test_headroom_invalid():
    with pytest.raises(ValueError):
        headroom_vs_target(projected_cost=0.0)
