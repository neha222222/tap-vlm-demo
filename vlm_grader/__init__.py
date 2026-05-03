"""vlm_grader — open-source VLM rubric grader for student artifacts.

Pre-application proof-of-concept for the TAP Cost-Efficient AI Model
project (DMP 2026, theapprenticeproject/C4GT_2026#2).

Modules:

- :mod:`rubric`: rubric definition + ``RubricScore`` dataclass.
- :mod:`prompt`: structured-output prompt builder.
- :mod:`schema`: JSON schema validation.
- :mod:`parser`: VLM output parser robust to leading/trailing prose.
- :mod:`metrics`: QWK + per-dimension MAE + format-validity rate.
- :mod:`inference`: VLM inference wrapper + ``MockVLM`` for tests/CI.
- :mod:`preprocessing`: image normalization + resize + OCR fallback hook.
- :mod:`cache`: response cache keyed by content hash.
- :mod:`cost`: cost-per-assessment calculator.
- :mod:`eval`: end-to-end eval harness.
- :mod:`synthetic`: synthetic dataset generator for tests.
"""

from vlm_grader.cache import ResponseCache
from vlm_grader.cost import CostModel, project_cost_per_assessment
from vlm_grader.eval import EvalReport, evaluate_grader
from vlm_grader.inference import MockVLM, VLMInterface
from vlm_grader.metrics import (
    format_validity_rate,
    per_dimension_mae,
    quadratic_weighted_kappa,
)
from vlm_grader.parser import parse_vlm_output
from vlm_grader.preprocessing import preprocess_image
from vlm_grader.prompt import build_prompt
from vlm_grader.rubric import (
    RUBRIC_DIMENSIONS,
    SCORE_DESCRIPTORS,
    RubricScore,
)
from vlm_grader.schema import JSON_SCHEMA, validate_against_schema
from vlm_grader.synthetic import SyntheticDataset, make_synthetic_artifact

__all__ = [
    "RUBRIC_DIMENSIONS",
    "SCORE_DESCRIPTORS",
    "RubricScore",
    "JSON_SCHEMA",
    "validate_against_schema",
    "build_prompt",
    "parse_vlm_output",
    "VLMInterface",
    "MockVLM",
    "preprocess_image",
    "ResponseCache",
    "CostModel",
    "project_cost_per_assessment",
    "quadratic_weighted_kappa",
    "per_dimension_mae",
    "format_validity_rate",
    "EvalReport",
    "evaluate_grader",
    "SyntheticDataset",
    "make_synthetic_artifact",
]

__version__ = "0.1.0"
