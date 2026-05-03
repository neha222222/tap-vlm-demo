"""LoRA fine-tuning scaffold.

Imported lazily — the heavy modules (torch, transformers, peft,
bitsandbytes) are not required to install or test ``vlm_grader``.
The training CLI lives in :mod:`vlm_grader.training.train` and the
hyperparameters in :mod:`vlm_grader.training.lora_config`.
"""
