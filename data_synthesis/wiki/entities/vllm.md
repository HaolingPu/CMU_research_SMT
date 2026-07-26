---
title: vLLM
type: entity
tags: [tool, inference, serving]
sources:
  - ../codes/gigaspeech/future_sampling/llm_future_sampling_final.py
created: 2026-06-01
updated: 2026-06-01
---

# vLLM

Inference engine used throughout, two ways: a **local** `LLM()` object with `SamplingParams` for
base-model future sampling, and **`vllm serve`** (OpenAI-compatible endpoint) for the instruct
translate/judge model. In [[future-sampling]] the base runs locally while the instruct model is
served on a separate GPU.

On the training/eval side it backs [[streaming-inference]] (v0.11.0, `tensor_parallel_size=2`,
prefix caching, `enforce_eager=True` to dodge a TP>1 shm_broadcast deadlock).

## Related
- [[future-sampling]], [[consensus-decoding]], [[streaming-inference]], [[infinisst-omni]].

## Sources
- code: `../codes/gigaspeech/future_sampling/llm_future_sampling_final.py`; `scripts/infer/infinisst_omni.py`
