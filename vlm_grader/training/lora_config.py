"""LoRA hyperparameters for VLM rubric-grader fine-tuning.

Validated to be loadable and serializable without the heavy training
dependencies installed. The real fine-tuning loop lives in
``vlm_grader.training.train`` and only imports torch/peft/transformers
on demand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class LoRAConfig:
    """LoRA fine-tuning hyperparameters.

    Defaults are calibrated for Qwen2-VL-2B-Instruct on a T4 GPU with
    NF4 4-bit quantization. The proposal commits to A/B these on a
    held-out 200-sample gold set.
    """

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Top-level training run configuration."""

    base_model: str = "Qwen/Qwen2-VL-2B-Instruct"
    output_dir: str = "checkpoints/qwen2vl-rubric-lora"
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    bf16: bool = False
    fp16: bool = True
    quantization: str = "nf4"  # "nf4" | "int8" | "none"
    max_seq_length: int = 1024
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    early_stopping_patience: int = 2
    seed: int = 42

    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps
