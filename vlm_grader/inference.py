"""VLM inference interface + a deterministic mock for tests / CI.

The real VLM (Qwen2-VL-2B-Instruct in 4-bit on a T4 GPU) is wired in
via :class:`Qwen2VLGrader` in ``vlm_grader.inference_qwen``, which is
imported lazily so this package can be installed and tested without
torch/transformers.
"""

from __future__ import annotations

import json
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Protocol


@dataclass
class GradingTrace:
    """The full record of one grading call.

    Carries enough information for cost analysis, audit, and eval.
    """

    raw_output: str
    latency_seconds: float
    model_id: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class _ImageLike(Protocol):
    pass


class VLMInterface(ABC):
    """Abstract interface every grader implementation satisfies."""

    model_id: str

    @abstractmethod
    def grade(self, image: Any, prompt: str) -> GradingTrace:
        """Return the raw VLM output for one ``(image, prompt)`` call."""


class MockVLM(VLMInterface):
    """Deterministic mock returning rubric-shaped JSON.

    Useful for testing the parser, eval harness, and cost calculator
    without a GPU. Behavior can be configured to:
    - Simulate noisy outputs (extra prose around the JSON).
    - Inject malformed JSON to test the parser's robustness.
    """

    def __init__(
        self,
        scores: Optional[List[dict]] = None,
        wrap_in_prose: bool = False,
        latency_seconds: float = 0.05,
        seed: Optional[int] = None,
        malformed_rate: float = 0.0,
    ) -> None:
        if malformed_rate < 0 or malformed_rate > 1:
            raise ValueError("malformed_rate must be in [0, 1]")
        self.model_id = "MockVLM-v1"
        self._scores = scores or [
            {
                "creativity": 3,
                "critical_thinking": 2,
                "problem_solving": 3,
                "agency": 2,
                "justification": "Default mock score.",
                "confidence": 0.7,
            }
        ]
        self._wrap = wrap_in_prose
        self._latency = float(latency_seconds)
        self._malformed_rate = float(malformed_rate)
        self._rng = random.Random(seed)
        self._idx = 0

    def grade(self, image: Any, prompt: str) -> GradingTrace:
        time.sleep(self._latency)
        score_dict = self._scores[self._idx % len(self._scores)]
        self._idx += 1
        if self._malformed_rate > 0 and self._rng.random() < self._malformed_rate:
            # Drop a required field, simulating a model failure.
            broken = dict(score_dict)
            broken.pop("creativity", None)
            raw_json = json.dumps(broken)
        else:
            raw_json = json.dumps(score_dict)
        if self._wrap:
            raw = f"Sure, here is the rubric assessment:\n\n{raw_json}\n\nLet me know if you need adjustments."
        else:
            raw = raw_json
        return GradingTrace(
            raw_output=raw,
            latency_seconds=self._latency,
            model_id=self.model_id,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(raw.split()),
        )
