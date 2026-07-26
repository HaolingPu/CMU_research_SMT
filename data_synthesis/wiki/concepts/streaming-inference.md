---
title: Streaming Inference
type: concept
tags: [inference, streaming, eval]
sources:
  - scripts/infer/infinisst_omni.py
  - scripts/infer/infer_slurm.sh
created: 2026-06-01
updated: 2026-06-01
---

# Streaming Inference

Real-time speech-to-text translation of trained checkpoints, via a `simuleval`
`SpeechToTextAgent` ([[simuleval]]) backed by [[vllm]]. The agent ([[infinisst-omni]],
`scripts/infer/infinisst_omni.py`) buffers audio in **960ms / 15360-sample** chunks and keeps a
multi-turn message history (`max_cache_chunks` 120 / `keep_cache_chunks` 60) for context.

Launchers `scripts/infer/infer_slurm{,_ja,_de}.sh` sweep 4 segment sizes (960/1920/2880/3840 ms).

## Per-language settings (do not unify)
`max-new-tokens` differs by language: **ja=30, de=60** per chunk. Prompt type is **Standard vs
EAST** — see [[east-prompt-handling]].

## Related
- [[checkpoint-evaluation]], [[latency-quality-tradeoff]], [[infinisst-omni]], [[simuleval]], [[vllm]], [[babel-cluster]].

## Sources
- code: `scripts/infer/infinisst_omni.py`, `scripts/infer/infer_slurm*.sh`
