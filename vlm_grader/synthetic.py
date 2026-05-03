"""Synthetic dataset generator.

Lets the test suite + CI exercise the full pipeline (preprocessing →
VLM → parser → eval) without any real images or VLM weights.

Each synthetic artifact is a deterministic placeholder paired with
a deterministic gold rubric score and a "VLM output" string that the
parser is expected to handle.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from vlm_grader.rubric import RubricScore


@dataclass
class SyntheticArtifact:
    artifact_id: str
    image_bytes: bytes  # placeholder: keeps the API signature image-shaped
    gold: RubricScore


@dataclass
class SyntheticDataset:
    artifacts: List[SyntheticArtifact]

    def __len__(self) -> int:
        return len(self.artifacts)

    def golds(self) -> List[RubricScore]:
        return [a.gold for a in self.artifacts]


def make_synthetic_artifact(idx: int, rng: random.Random) -> SyntheticArtifact:
    """Make one deterministic synthetic artifact."""
    creativity = rng.randint(1, 4)
    critical_thinking = rng.randint(1, 4)
    problem_solving = rng.randint(1, 4)
    agency = rng.randint(1, 4)
    gold = RubricScore(
        creativity=creativity,
        critical_thinking=critical_thinking,
        problem_solving=problem_solving,
        agency=agency,
        justification=f"Synthetic gold for artifact {idx}.",
    )
    return SyntheticArtifact(
        artifact_id=f"synthetic-{idx:04d}",
        image_bytes=f"image-{idx}".encode("utf-8"),
        gold=gold,
    )


def make_synthetic_dataset(n: int = 100, seed: int = 0) -> SyntheticDataset:
    rng = random.Random(seed)
    return SyntheticDataset(
        artifacts=[make_synthetic_artifact(i, rng) for i in range(n)]
    )


def make_mock_vlm_outputs(
    dataset: SyntheticDataset,
    *,
    error_dimension_drift: int = 0,
    malformed_indices: Optional[List[int]] = None,
) -> List[str]:
    """Produce deterministic raw VLM output strings for a dataset.

    Parameters
    ----------
    dataset:
        The synthetic dataset.
    error_dimension_drift:
        How much (-3 to +3) to bias each predicted dimension score
        relative to gold. 0 -> perfect agreement, 1 -> simulate a
        slightly-optimistic model.
    malformed_indices:
        Indices in the dataset whose VLM output should be malformed
        (drops a required field). Useful to test the parser's failure
        path.
    """
    malformed = set(malformed_indices or [])
    out: List[str] = []
    for i, art in enumerate(dataset.artifacts):
        if i in malformed:
            out.append("not even json haha")
            continue
        biased = {
            "creativity": _clamp(art.gold.creativity + error_dimension_drift, 1, 4),
            "critical_thinking": _clamp(
                art.gold.critical_thinking + error_dimension_drift, 1, 4
            ),
            "problem_solving": _clamp(
                art.gold.problem_solving + error_dimension_drift, 1, 4
            ),
            "agency": _clamp(art.gold.agency + error_dimension_drift, 1, 4),
            "justification": f"Mock prediction for {art.artifact_id}.",
            "confidence": 0.7,
        }
        out.append(json.dumps(biased))
    return out


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def synthetic_artifacts_for_eval(
    n: int = 100,
    seed: int = 0,
) -> List[Tuple[bytes, str, RubricScore]]:
    """Build a list of ``(image_bytes, prompt, gold)`` for the eval harness."""
    ds = make_synthetic_dataset(n=n, seed=seed)
    return [(a.image_bytes, "synthetic prompt", a.gold) for a in ds.artifacts]
