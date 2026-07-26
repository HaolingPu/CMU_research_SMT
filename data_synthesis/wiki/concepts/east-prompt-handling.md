---
title: EAST vs Standard Prompt Handling
type: concept
tags: [inference, prompt, gotcha]
sources:
  - scripts/infer/infinisst_omni.py
created: 2026-06-01
updated: 2026-06-01
---

# EAST vs Standard Prompt Handling

[[infinisst-omni]] (`scripts/infer/infinisst_omni.py`) supports two prompt types:

- **EAST** (line ~66): includes an explicit latency annotation (low/medium/high), selected via
  `--EAST-latency-type`.
- **Standard** (line ~67): same instruction without the latency clause.

## The gotcha
Several zh checkpoints that are *named* EAST-style (e.g. **EAST-latency2mult**) and the
**Simul-MuST-C** models were actually **trained on the Standard prompt**. They must be inferred
with `--prompt-type Standard`, **not** the EAST default — otherwise quality drops. Match the
infer prompt to the training prompt, not the model's name.

## Related
- [[east]], [[streaming-inference]], [[checkpoint-evaluation]].

## Sources
- code: `scripts/infer/infinisst_omni.py` (lines ~65-68); `scripts/train/train_EAST_s.sh`
