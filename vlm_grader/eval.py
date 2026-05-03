"""End-to-end evaluation harness.

Given a :class:`VLMInterface` and a list of (image, prompt, gold-score)
triples, runs the grader, parses outputs, and computes:

- Format-validity rate (does the output parse?).
- Quadratic Weighted Kappa per dimension vs gold.
- Per-dimension MAE.
- Subset accuracy (exact match across all four dimensions).
- Mean / p50 / p95 latency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from vlm_grader.cost import CostModel, project_cost_per_assessment
from vlm_grader.inference import VLMInterface
from vlm_grader.metrics import (
    format_validity_rate,
    per_dimension_mae,
    quadratic_weighted_kappa,
    subset_accuracy,
)
from vlm_grader.parser import ParseError, parse_vlm_output
from vlm_grader.rubric import RUBRIC_DIMENSIONS, RubricScore


@dataclass
class EvalReport:
    n_artifacts: int
    format_validity_rate: float
    qwk_per_dimension: Dict[str, float]
    mae_per_dimension: Dict[str, float]
    subset_accuracy: float
    mean_latency_seconds: float
    p95_latency_seconds: float
    projected_cost_per_assessment_inr: float
    raw_outputs: List[str] = field(default_factory=list)
    failures: List[Tuple[int, str]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"n = {self.n_artifacts}",
            f"format_validity = {self.format_validity_rate:.2%}",
            f"subset_accuracy = {self.subset_accuracy:.2%}",
            f"mean latency = {self.mean_latency_seconds:.3f}s",
            f"p95 latency  = {self.p95_latency_seconds:.3f}s",
            f"projected cost/assessment = ₹{self.projected_cost_per_assessment_inr:.4f}",
            "QWK per dimension:",
        ]
        for dim, k in self.qwk_per_dimension.items():
            lines.append(f"  {dim:20s} {k:.3f}")
        lines.append("MAE per dimension:")
        for dim, m in self.mae_per_dimension.items():
            lines.append(f"  {dim:20s} {m:.3f}")
        return "\n".join(lines)


def evaluate_grader(
    grader: VLMInterface,
    artifacts: Iterable[Tuple[Any, str, RubricScore]],
    cost_model: Optional[CostModel] = None,
) -> EvalReport:
    """Run ``grader`` over ``artifacts`` and produce an :class:`EvalReport`.

    Parameters
    ----------
    grader:
        Any :class:`VLMInterface` implementation.
    artifacts:
        Iterable of ``(image, prompt, gold_RubricScore)`` triples.
    cost_model:
        Optional pricing model (defaults to :class:`CostModel`).
    """
    artifacts = list(artifacts)
    if not artifacts:
        raise ValueError("artifacts must be non-empty")

    raw_outputs: List[str] = []
    latencies: List[float] = []
    parsed: List[Optional[RubricScore]] = []
    failures: List[Tuple[int, str]] = []
    gold: List[RubricScore] = []

    for idx, (image, prompt, gold_score) in enumerate(artifacts):
        gold.append(gold_score)
        trace = grader.grade(image, prompt)
        raw_outputs.append(trace.raw_output)
        latencies.append(trace.latency_seconds)
        try:
            parsed.append(parse_vlm_output(trace.raw_output))
        except ParseError as exc:
            parsed.append(None)
            failures.append((idx, str(exc)))

    rate, _ = format_validity_rate(raw_outputs)
    valid_pairs = [(p, g) for p, g in zip(parsed, gold) if p is not None]
    if not valid_pairs:
        # No valid predictions — empty metrics.
        return EvalReport(
            n_artifacts=len(artifacts),
            format_validity_rate=rate,
            qwk_per_dimension={d: 0.0 for d in RUBRIC_DIMENSIONS},
            mae_per_dimension={d: 0.0 for d in RUBRIC_DIMENSIONS},
            subset_accuracy=0.0,
            mean_latency_seconds=sum(latencies) / len(latencies),
            p95_latency_seconds=_percentile(latencies, 95),
            projected_cost_per_assessment_inr=_project_cost_for_latency(
                sum(latencies) / len(latencies), cost_model=cost_model
            ),
            raw_outputs=raw_outputs,
            failures=failures,
        )

    valid_preds = [p for p, _ in valid_pairs]
    valid_golds = [g for _, g in valid_pairs]

    qwk = {
        dim: quadratic_weighted_kappa(
            [getattr(p, dim) for p in valid_preds],
            [getattr(g, dim) for g in valid_golds],
        )
        for dim in RUBRIC_DIMENSIONS
    }
    mae = per_dimension_mae(valid_preds, valid_golds)
    subset = subset_accuracy(valid_preds, valid_golds)
    mean_latency = sum(latencies) / len(latencies)

    return EvalReport(
        n_artifacts=len(artifacts),
        format_validity_rate=rate,
        qwk_per_dimension=qwk,
        mae_per_dimension=mae,
        subset_accuracy=subset,
        mean_latency_seconds=mean_latency,
        p95_latency_seconds=_percentile(latencies, 95),
        projected_cost_per_assessment_inr=_project_cost_for_latency(
            mean_latency, cost_model=cost_model
        ),
        raw_outputs=raw_outputs,
        failures=failures,
    )


def _project_cost_for_latency(
    mean_latency_seconds: float,
    cost_model: Optional[CostModel] = None,
) -> float:
    """Return zero cost for zero-latency mocks, otherwise use the cost model."""
    if mean_latency_seconds <= 0:
        return 0.0
    return project_cost_per_assessment(mean_latency_seconds, cost_model=cost_model)


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)
