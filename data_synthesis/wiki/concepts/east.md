---
title: EAST
type: concept
tags: [synthesis, segmentation, latency]
sources:
  - ../codes/gigaspeech/llm_output_gigaspeech_trajectory.py
  - ../codes/gigaspeech/east/pipeline.sh
created: 2026-06-01
updated: 2026-06-01
---

# EAST

Baseline simultaneous-translation synthesis method (no future sampling). An instruct LLM
segments the full source into three latency levels — **low / medium / high** — each with a
pre-segmented target, via `../codes/gigaspeech/llm_output_gigaspeech_trajectory.py` (Qwen3-30B
instruct over [[vllm]], chat-mode with a few-shot example, JSON guided decoding). Runs through
the shared [[synthesis-pipeline]]. A `refined_east` variant uses an improved prompt + trajectory
filtering.

The zh latency-stratified training split is ~12.5K (4166 / 4166 / 4168 over low/med/high).

## Inference-time gotcha
zh EAST-style trained checkpoints (e.g. EAST-latency2mult) are **trained on the Standard prompt**
and must be inferred with `--prompt-type Standard`, not the EAST default. See
[[east-prompt-handling]].

## Related
- [[salami]] (alternative segmentation format), [[synthesis-pipeline]],
  [[future-sampling]] (the online successor), [[la-n-vs-wait-k]].

## Sources
- code: `../codes/gigaspeech/llm_output_gigaspeech_trajectory.py`, `east/pipeline.sh`
