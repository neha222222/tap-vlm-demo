"""TAP 21st-century-skills rubric definition + JSON output schema.

Mirrors the rubric framework referenced in the C4GT 2026 issue: scoring
student artifacts on creativity, critical thinking, problem-solving, and
agency on a 1-4 ordinal scale.

Kept as a single-file module so the Colab notebook can run in one cell
without external setup.
"""

from __future__ import annotations

import json
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


@dataclass
class RubricScore:
    """One rubric assessment for a single student artifact."""

    creativity: int
    critical_thinking: int
    problem_solving: int
    agency: int
    justification: str
    confidence: Optional[float] = None  # 0.0 to 1.0, optional model-self-reported

    def __post_init__(self) -> None:
        for dim in RUBRIC_DIMENSIONS:
            value = getattr(self, dim)
            if not isinstance(value, int) or not 1 <= value <= 4:
                raise ValueError(
                    f"{dim} must be int in [1, 4], got {value!r}"
                )
        if not isinstance(self.justification, str) or not self.justification.strip():
            raise ValueError("justification must be a non-empty string")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        # Cap justification length to keep tokens predictable.
        if len(self.justification) > 280:
            raise ValueError(
                f"justification too long ({len(self.justification)} chars, max 280)"
            )

    def to_dict(self) -> dict:
        return {
            "creativity": self.creativity,
            "critical_thinking": self.critical_thinking,
            "problem_solving": self.problem_solving,
            "agency": self.agency,
            "justification": self.justification,
            "confidence": self.confidence,
        }


# JSON schema for the structured-output decoder. Used both as
# documentation and as a validation contract for VLM outputs.
JSON_SCHEMA = {
    "type": "object",
    "required": [
        "creativity",
        "critical_thinking",
        "problem_solving",
        "agency",
        "justification",
    ],
    "additionalProperties": False,
    "properties": {
        "creativity": {"type": "integer", "minimum": 1, "maximum": 4},
        "critical_thinking": {"type": "integer", "minimum": 1, "maximum": 4},
        "problem_solving": {"type": "integer", "minimum": 1, "maximum": 4},
        "agency": {"type": "integer", "minimum": 1, "maximum": 4},
        "justification": {"type": "string", "minLength": 1, "maxLength": 280},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


def parse_vlm_output(text: str) -> RubricScore:
    """Parse a raw VLM string response into a RubricScore.

    Tolerant of leading/trailing prose around the JSON block — finds the
    first ``{`` and last ``}`` and parses the slice between.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object found in VLM output: {text!r}")
    blob = text[start : end + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in VLM output: {exc}") from exc
    return RubricScore(
        creativity=int(data["creativity"]),
        critical_thinking=int(data["critical_thinking"]),
        problem_solving=int(data["problem_solving"]),
        agency=int(data["agency"]),
        justification=str(data["justification"]),
        confidence=(
            float(data["confidence"]) if data.get("confidence") is not None else None
        ),
    )


def build_prompt(rubric_descriptors: Optional[Dict[int, str]] = None) -> str:
    """Construct the rubric-grading instruction for the VLM."""
    descriptors = rubric_descriptors or SCORE_DESCRIPTORS
    descriptor_block = "\n".join(
        f"  {score} — {desc}" for score, desc in descriptors.items()
    )
    return (
        "You are a rubric grader for student artifacts (drawings, prototypes, "
        "or written responses). Score the attached artifact on the four "
        "21st-century-skill dimensions below using the 1-4 scale, where:\n"
        f"{descriptor_block}\n\n"
        "Dimensions:\n"
        "  - creativity: originality and inventive elements.\n"
        "  - critical_thinking: reasoning, analysis, evaluation of ideas.\n"
        "  - problem_solving: identifies a goal and works toward it.\n"
        "  - agency: takes initiative, makes choices, owns the outcome.\n\n"
        "Respond with a single JSON object matching this exact schema, and "
        "nothing else:\n"
        "{\n"
        '  "creativity": <int 1-4>,\n'
        '  "critical_thinking": <int 1-4>,\n'
        '  "problem_solving": <int 1-4>,\n'
        '  "agency": <int 1-4>,\n'
        '  "justification": "<one sentence, max 280 chars>",\n'
        '  "confidence": <float 0-1>\n'
        "}"
    )


def quadratic_weighted_kappa(
    rater_a: List[int], rater_b: List[int], min_score: int = 1, max_score: int = 4
) -> float:
    """Quadratic Weighted Kappa for two ordinal raters.

    Standard inter-rater agreement metric for ordinal rubrics. QWK ranges
    from -1 (perfect disagreement) through 0 (chance) to 1 (perfect
    agreement). Implemented from scratch (no sklearn dep) so the Colab
    notebook stays light.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("rater_a and rater_b must be the same length")
    if not rater_a:
        raise ValueError("inputs must be non-empty")
    n_categories = max_score - min_score + 1

    # Confusion matrix
    O = [[0] * n_categories for _ in range(n_categories)]
    for a, b in zip(rater_a, rater_b):
        O[a - min_score][b - min_score] += 1

    # Marginal histograms
    hist_a = [sum(row) for row in O]
    hist_b = [sum(O[i][j] for i in range(n_categories)) for j in range(n_categories)]
    n = sum(hist_a)

    # Quadratic weight matrix
    W = [
        [((i - j) ** 2) / ((n_categories - 1) ** 2) for j in range(n_categories)]
        for i in range(n_categories)
    ]

    # Expected matrix under independence
    E = [[(hist_a[i] * hist_b[j]) / n for j in range(n_categories)] for i in range(n_categories)]

    num = sum(W[i][j] * O[i][j] for i in range(n_categories) for j in range(n_categories))
    den = sum(W[i][j] * E[i][j] for i in range(n_categories) for j in range(n_categories))
    if den == 0:
        return 1.0 if num == 0 else 0.0
    return 1.0 - num / den
