"""Qwen2-VL-2B-Instruct grader (lazy import of torch / transformers).

Imported only when explicitly requested — keeps the package light
enough to install + test without GPU dependencies. The Colab
notebook imports this module directly and the GPU stack lives there.
"""

from __future__ import annotations

import time
from typing import Any

from vlm_grader.inference import GradingTrace, VLMInterface


class Qwen2VLGrader(VLMInterface):
    """Real VLM grader backed by Qwen2-VL-2B-Instruct in 4-bit."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        load_in_4bit: bool = True,
        max_new_tokens: int = 200,
        device_map: str = "auto",
    ) -> None:
        try:
            import torch
            from transformers import (  # type: ignore
                AutoProcessor,
                BitsAndBytesConfig,
                Qwen2VLForConditionalGeneration,
            )
        except ImportError as exc:  # pragma: no cover - GPU-only path
            raise ImportError(
                "Qwen2VLGrader requires torch + transformers + bitsandbytes. "
                "Install them with `pip install torch transformers accelerate "
                "bitsandbytes pillow`."
            ) from exc

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
        )
        self._model.eval()
        self._torch = torch

    def grade(self, image: Any, prompt: str) -> GradingTrace:
        torch = self._torch
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=[chat], images=[image], return_tensors="pt").to(
            self._model.device
        )
        prompt_tokens = int(inputs["input_ids"].shape[1])

        t0 = time.time()
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        elapsed = time.time() - t0

        completion_ids = out[:, inputs["input_ids"].shape[1] :]
        raw = self._processor.batch_decode(completion_ids, skip_special_tokens=True)[0]
        return GradingTrace(
            raw_output=raw,
            latency_seconds=elapsed,
            model_id=self.model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(completion_ids.shape[1]),
        )
