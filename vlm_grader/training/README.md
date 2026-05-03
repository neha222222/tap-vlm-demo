# Training scaffold

The fine-tuning loop is *scaffolded* in this demo — it documents the
intended pipeline and the hyperparameters in
[`lora_config.py`](./lora_config.py), but the actual training run is
the body of the DMP work itself.

The real `train.py` (added during the program) loads:

```
Qwen/Qwen2-VL-2B-Instruct
  + bitsandbytes NF4 quantization (4-bit)
  + LoRA adapters (r=16, alpha=32, target=q_proj/k_proj/v_proj/o_proj)
  + supervised fine-tuning on TAP's rubric-graded artifact dataset
```

Notes for upstream:

- LoRA adapters live in `target_modules` — narrow set chosen so the
  vision encoder stays frozen and only the language head learns
  rubric-aligned generation.
- bitsandbytes NF4 + double-quantization keeps VRAM under ~3 GB on a
  T4.
- AWQ or GPTQ post-training quantization gives faster inference
  throughput than QLoRA at serve time — we apply it after fine-tuning
  finishes.
- Loss is standard causal LM loss on the structured-JSON target.
  Optional auxiliary rubric-distance loss penalizes large numeric
  score deviations more than format errors; toggleable via a flag.

`LoRAConfig` and `TrainingConfig` are importable + serializable
without torch installed, so they can be tested in CI.
