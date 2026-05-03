"""Structured-output prompt builder.

Produces the rubric-grading instruction the VLM sees at inference time.
Kept small so it can be tweaked + A/B-tested cheaply.
"""

from __future__ import annotations

from typing import Dict, Optional

from vlm_grader.rubric import RUBRIC_DIMENSIONS, SCORE_DESCRIPTORS


DIMENSION_DESCRIPTIONS: Dict[str, str] = {
    "creativity": "originality and inventive elements visible in the artifact.",
    "critical_thinking": "reasoning, analysis, evaluation of ideas.",
    "problem_solving": "identifies a goal or problem and works toward solving it.",
    "agency": "takes initiative, makes choices, owns the outcome.",
}


def build_prompt(
    rubric_descriptors: Optional[Dict[int, str]] = None,
    artifact_type_hint: Optional[str] = None,
) -> str:
    """Construct the rubric-grading instruction for the VLM.

    Parameters
    ----------
    rubric_descriptors:
        Override the default 1-4 descriptors. Useful for A/B testing
        prompt variants.
    artifact_type_hint:
        Optional one-line cue (e.g. "this is a student-drawn poster").
        Improves zero-shot accuracy modestly.
    """
    descriptors = rubric_descriptors or SCORE_DESCRIPTORS
    descriptor_block = "\n".join(
        f"  {score} — {desc}" for score, desc in descriptors.items()
    )
    dimensions_block = "\n".join(
        f"  - {dim}: {DIMENSION_DESCRIPTIONS[dim]}" for dim in RUBRIC_DIMENSIONS
    )
    hint_block = (
        f"Artifact type hint: {artifact_type_hint}\n\n"
        if artifact_type_hint
        else ""
    )
    return (
        "You are a rubric grader for student artifacts (drawings, "
        "prototypes, or written responses). Score the attached "
        "artifact on the four 21st-century-skill dimensions below "
        "using the 1-4 scale, where:\n"
        f"{descriptor_block}\n\n"
        f"{hint_block}"
        "Dimensions:\n"
        f"{dimensions_block}\n\n"
        "Respond with a single JSON object matching this exact schema, "
        "and nothing else:\n"
        "{\n"
        '  "creativity": <int 1-4>,\n'
        '  "critical_thinking": <int 1-4>,\n'
        '  "problem_solving": <int 1-4>,\n'
        '  "agency": <int 1-4>,\n'
        '  "justification": "<one sentence, max 280 chars>",\n'
        '  "confidence": <float 0-1>\n'
        "}"
    )
