"""Tests for the LoRA training config dataclasses.

The actual training loop requires torch + peft + transformers; these
tests cover only the config classes, which are importable and
serializable without those deps.
"""

from dataclasses import asdict

from vlm_grader.training.lora_config import LoRAConfig, TrainingConfig


def test_default_lora_config():
    c = LoRAConfig()
    assert c.r == 16
    assert c.alpha == 32
    assert "q_proj" in c.target_modules


def test_default_training_config():
    t = TrainingConfig()
    assert t.base_model == "Qwen/Qwen2-VL-2B-Instruct"
    assert t.effective_batch_size() == t.per_device_train_batch_size * t.gradient_accumulation_steps


def test_lora_config_serializable():
    c = LoRAConfig()
    d = asdict(c)
    assert d["r"] == 16
    assert isinstance(d["target_modules"], list)


def test_training_config_serializable():
    t = TrainingConfig()
    d = asdict(t)
    assert d["learning_rate"] == 2e-4
    assert d["quantization"] == "nf4"
