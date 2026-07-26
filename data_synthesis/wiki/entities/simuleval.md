---
title: SimulEval
type: entity
tags: [tool, inference, eval]
sources:
  - scripts/infer/infinisst_omni.py
created: 2026-06-01
updated: 2026-06-01
---

# SimulEval

Simultaneous speech-to-text evaluation harness. Provides the `SpeechToTextAgent` interface that
[[infinisst-omni]] implements, drives chunk-wise audio streaming, and emits `instances.log`
consumed by [[checkpoint-evaluation]]. Latency metrics (LongYAAL / StreamLAAL) come from its
eval side (omnisteval longform wrapper).

## Related
- [[streaming-inference]], [[checkpoint-evaluation]], [[infinisst-omni]].

## Sources
- code: `scripts/infer/infinisst_omni.py`, `scripts/infer/eval_all_ckpts.sh`
