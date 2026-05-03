"""Cost-per-assessment calculator.

Closes the unit-economics question the TAP issue makes the headline
constraint: can we deliver rubric grading at <₹0.10 per assessment?
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    """Pricing assumptions for the inference deployment.

    Defaults reflect typical Indian cloud prices for a serverless T4
    GPU as of 2025-2026.
    """

    gpu_hourly_inr: float = 50.0
    overhead_factor: float = 1.2  # +20% for orchestration / cold starts
    cache_hit_rate: float = 0.0   # 0 = no caching, 1 = all hits free


def project_cost_per_assessment(
    mean_latency_seconds: float,
    batch_size: int = 1,
    cost_model: CostModel = None,
) -> float:
    """Project per-assessment INR cost given measured latency.

    Parameters
    ----------
    mean_latency_seconds:
        End-to-end latency of one inference.
    batch_size:
        How many assessments are graded per inference call. Reduces
        per-assessment cost by amortizing GPU time.
    cost_model:
        Pricing assumptions (defaults to :class:`CostModel`).

    Returns
    -------
    inr:
        Projected cost per assessment in INR. Includes the
        ``cache_hit_rate`` discount (cached calls are free).
    """
    if mean_latency_seconds <= 0:
        raise ValueError("mean_latency_seconds must be > 0")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    cost_model = cost_model or CostModel()
    if not 0.0 <= cost_model.cache_hit_rate <= 1.0:
        raise ValueError("cache_hit_rate must be in [0, 1]")

    miss_fraction = 1.0 - cost_model.cache_hit_rate
    seconds_per_inference = mean_latency_seconds * cost_model.overhead_factor
    seconds_per_assessment = seconds_per_inference / batch_size
    cost_per_assessment_uncached = (seconds_per_assessment / 3600.0) * cost_model.gpu_hourly_inr
    return cost_per_assessment_uncached * miss_fraction


def headroom_vs_target(projected_cost: float, target_inr: float = 0.10) -> float:
    """Return ``target_inr / projected_cost`` — the headroom factor.

    Values >1 mean the projection is cheaper than target.
    """
    if projected_cost <= 0:
        raise ValueError("projected_cost must be > 0")
    return target_inr / projected_cost
