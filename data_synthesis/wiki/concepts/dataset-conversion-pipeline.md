---
title: Dataset Conversion (→ SWIFT/Megatron)
type: concept
tags: [training, data]
sources:
  - scripts/train/convert2swift.py
  - scripts/train/convert2swift_LA.py
  - scripts/train/convert2swift_consensus.py
created: 2026-06-01
updated: 2026-06-26
---

# Dataset Conversion (→ SWIFT/Megatron)

Turns synthesis latency-trajectory JSONLs into SWIFT/[[megatron-swift]] training instances:
trajectories → chunked audio WAVs (**15360 samples / 960ms @ 16kHz**) → multi-turn messages with
latency-aware instructions. Base manifest: GigaSpeech XL ([[gigaspeech]]).

Per-policy converters in `scripts/train/`:
- `convert2swift.py`, `convert2swift_east-mult.py` — [[east]] trajectories.
- `convert2swift_LA.py`, `convert2swift_PA.py` — local-agreement / prefix-alignment; support
  random chunk-size multipliers (1–12×) to simulate latency regimes ([[la-n-vs-wait-k]]).
- `convert2swift_consensus.py` — [[consensus-decoding]] outputs. Reads a flat dir of survivor
  `<utt_id>.json`, draws a random chunk multiplier (1–12×) per utt with a **fixed seed (42)** in
  sorted-glob order, `SAMPLE_N` downsamples to a fixed count. Determinism note: it never adds/removes
  `target_trajectory` deltas, so two input dirs with the same basenames yield the **identical
  instance set + multipliers** — exploited to build the clean period-fix ablation in
  [[2026-06-qwenasr-asr-regression-periodfix]] (same 12,446 instances; only the 。 placement differs).
- `convert2swift_simul-mustc.py` — Simul-MuST-C chunked outputs (no latency annotation).

## Related
- [[megatron-swift]], [[gigaspeech]], [[synthesis-pipeline]], [[la-n-vs-wait-k]],
  [[2026-06-qwenasr-asr-regression-periodfix]].

## Sources
- code: `scripts/train/convert2swift_*.py`
