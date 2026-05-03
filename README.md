# TAP VLM Rubric Grader Demo — Proof of Concept

Pre-application proof-of-concept for the [TAP Cost-Efficient AI Model
project](https://github.com/theapprenticeproject/C4GT_2026/issues/2) in
DMP 2026.

A small, audit-friendly rubric-grading pipeline for benchmarking
rubric-based VLM assessment. It includes a structured-output schema,
robust JSON parsing, preprocessing, cache-aware inference stubs,
cost projection, synthetic evaluation data, Quadratic Weighted Kappa
metrics, and a LoRA fine-tuning config sketch for Qwen2-VL.

## What's here

```
vlm_grader/
├── rubric.py          # rubric dimensions and validation
├── schema.py          # structured output JSON schema
├── parser.py          # robust parser for prose + JSON VLM output
├── eval.py            # end-to-end grader evaluation harness
├── metrics.py         # QWK, MAE, format validity, subset accuracy
├── cost.py            # cost-per-assessment projection
├── cache.py           # content-addressed result cache
├── preprocessing.py   # image resizing / normalization helpers
├── inference.py       # VLM interface + deterministic MockVLM
├── inference_qwen.py  # optional Qwen2-VL adapter sketch
└── training/
    └── lora_config.py # LoRA fine-tuning config skeleton
tests/                 # 88 fast unit tests
notebook.ipynb         # Colab-ready end-to-end demo on a T4 GPU
.github/workflows/test.yml  # CI matrix on Python 3.10 / 3.11 / 3.12
```

## What this proves

1. The rubric-output contract (4 dimensions x 1-4 ordinal score + short justification) round-trips cleanly through `parse_vlm_output`.
2. Format validity, QWK, MAE, subset accuracy, latency, and projected cost are measurable in one evaluation report.
3. The cost model makes the issue's INR 0.10 target explicit and testable with batching and cache-hit assumptions.
4. The package separates production concerns: schema, parsing, inference, cache, preprocessing, metrics, and training config.

## Run the tests

```bash
pip install pytest
pytest tests/ -q
```

Expected: 88 passed in <1s.

## Run the notebook

Open `notebook.ipynb` in Google Colab. Switch the runtime to T4 GPU. Run all cells. First run takes ~4 min for the model download.

The notebook walks through:

1. Loading Qwen2-VL-2B-Instruct in 4-bit (~3 GB VRAM).
2. Building the rubric prompt + structured JSON schema.
3. Inference on two public-domain student-drawing samples.
4. Parsing + format-validity measurement.
5. Cost-per-assessment projection against the INR 0.10 target.
6. QWK metric on a synthetic evaluation set.

## What this notebook does NOT do

Honest scoping for the mentor:
- No fine-tuning yet — that's the DMP work itself.
- No batched inference — single-sample only, so cost numbers are an upper bound.
- No human gold labels for the sample images — QWK tests use synthetic ratings.
- No constrained decoding (`outlines` / `xgrammar`) — prompt-only structured output, which is exactly why format-validity is <100 % zero-shot.

## License

MIT.
