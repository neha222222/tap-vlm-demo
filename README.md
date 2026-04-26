# TAP VLM Rubric Grader Demo — Proof of Concept

Pre-application proof-of-concept for the [TAP Cost-Efficient AI Model
project](https://github.com/theapprenticeproject/C4GT_2026/issues/2) in
DMP 2026.

A small, audit-friendly rubric-grading pipeline running zero-shot
**Qwen2-VL-2B-Instruct** in 4-bit quantization, with a structured-output
schema, a parser that's robust to leading prose, and a Quadratic
Weighted Kappa metric for ordinal inter-rater agreement.

## What's here

```
rubric.py              # rubric definition, JSON schema, parser, QWK metric
test_rubric.py         # 16 unit tests (run with `pytest`)
notebook.ipynb         # Colab-ready end-to-end demo on a T4 GPU
.github/workflows/test.yml  # CI matrix on Python 3.10 / 3.11 / 3.12
```

## What this proves

1. The rubric-output contract (4 dimensions × 1-4 ordinal score + ≤280-char justification) round-trips cleanly through `parse_vlm_output`.
2. Format validity is measurable end-to-end — a baseline number for the proposal's fine-tuning effort to improve on.
3. QWK (the proposal's primary metric vs human gold) is implemented from scratch in <50 lines of Python — no sklearn dep.
4. Cost projection on a T4 (~₹50/hour) is wired into the notebook, surfacing the gap from the zero-shot baseline to the ₹0.10/assessment target.

## Run the tests

```bash
pip install pytest
pytest test_rubric.py -v
```

Expected: 16 passed in <1s.

## Run the notebook

Open `notebook.ipynb` in Google Colab. Switch the runtime to T4 GPU. Run all cells. First run takes ~4 min for the model download.

The notebook walks through:

1. Loading Qwen2-VL-2B-Instruct in 4-bit (~3 GB VRAM).
2. Building the rubric prompt + structured JSON schema.
3. Inference on two public-domain student-drawing samples.
4. Parsing + format-validity rate measurement.
5. Cost-per-assessment projection.
6. QWK metric on a synthetic 8-artifact eval.

## What this notebook does NOT do

Honest scoping for the mentor:
- No fine-tuning yet — that's the DMP work itself.
- No batched inference — single-sample only, so cost numbers are an upper bound.
- No human gold labels for the sample images — QWK demo uses synthetic ratings.
- No constrained decoding (`outlines` / `xgrammar`) — prompt-only structured output, which is exactly why format-validity is <100 % zero-shot.

Each gap maps to a milestone in the [DMP proposal](../../proposals/01_TAP_VLM_Eval_Proposal.md)'s 12-week timeline.

## License

MIT.
