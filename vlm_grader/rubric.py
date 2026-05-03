"""Rubric definition + ``RubricScore`` dataclass.

The rubric matches the framework referenced in TAP's DMP 2026 issue:
scoring student artifacts on creativity, critical thinking,
problem-solving, and agency on a 1-4 ordinal scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


RUBRIC_DIMENSIONS: List[str] = [
    "creativity",
    "critical_thinking",
    "problem_solving",
    "agency",
]


# 1 = Emerging, 2 = Developing, 3 = Proficient, 4 = Advanced
SCORE_DESCRIPTORS: Dict[int, str] = {
    1: "Emerging — minimal evidence of the skill in the artifact.",
    2: "Developing — partial evidence; basic effort but unrefined.",
    3: "Proficient — clear evidence; meets expectations for the skill.",
    4: "Advanced — strong evidence; exceeds expectations, original thought.",
}

MAX_JUSTIFICATION_LEN = 280


@dataclass
class RubricScore:
    """One rubric assessment for a single student artifact.

    All four dimensions are required and must be integers in [1, 4].
    The justification is a one-sentence explanation, capped to keep
    token usage predictable.
    """

    creativity: int
    critical_thinking: int
    problem_solving: int
    agency: int
    justification: str
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        for dim in RUBRIC_DIMENSIONS:
            value = getattr(self, dim)
            if not isinstance(value, int) or not 1 <= value <= 4:
                raise ValueError(f"{dim} must be int in [1, 4], got {value!r}")
        if not isinstance(self.justification, str) or not self.justification.strip():
            raise ValueError("justification must be a non-empty string")
        if len(self.justification) > MAX_JUSTIFICATION_LEN:
            raise ValueError(
                f"justification too long ({len(self.justification)} chars, "
                f"max {MAX_JUSTIFICATION_LEN})"
            )
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def to_dict(self) -> dict:
        return {
            "creativity": self.creativity,
            "critical_thinking": self.critical_thinking,
            "problem_solving": self.problem_solving,
            "agency": self.agency,
            "justification": self.justification,
            "confidence": self.confidence,
        }

    def overall(self) -> float:
        """Mean of the four rubric scores."""
        return (
            self.creativity
            + self.critical_thinking
            + self.problem_solving
            + self.agency
        ) / 4.0
