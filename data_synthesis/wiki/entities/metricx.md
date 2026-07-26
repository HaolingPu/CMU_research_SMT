---
title: MetricX (QE)
type: entity
tags: [tool, eval, qe]
sources:
  - ../codes/gigaspeech/convert_metricx_gigaspeech.py
  - ../codes/gigaspeech/filter_metricx_gigaspeech.py
  - ../codes/metricx/
created: 2026-06-01
updated: 2026-06-26
---

# MetricX (QE)

Reference-free quality-estimation model used to filter synthesis outputs. Model
`metricx-24-hybrid-xl-v2p6` with the `mt5-xl` tokenizer (conda env `metricx`). Lower score = better.

In the [[synthesis-pipeline]]: `convert_metricx_gigaspeech.py` builds the input JSONL (auto-detects
latency level, language-aware joining, 8-shard split) → 8-GPU QE predict → `filter_metricx_gigaspeech.py`
keeps entries with prediction **≤ 3.0**.

Also drives ablation scoring in [[min-p-sampling]] sweeps. As an eval metric it complements
COMET/BLEU — see [[comet-vs-bleu-ranking]].

**Per-sentence QE-MAX** (drop an utterance if *any* sub-sentence > threshold) is the consensus
filter, applied after [[segale-alignment]] in `submit_J40k_post.sh`. On the new qwenasr data it
passes ~43% (17,089/40,000) and is the single biggest lever rescuing the regressed run (collapsed
seg960 BLEU 6.9 → ~30) — but it does **not** close the old↔new ASR data-quality gap. See
[[2026-06-qwenasr-asr-regression-periodfix]].

## Related
- [[synthesis-pipeline]], [[comet-vs-bleu-ranking]], [[min-p-sampling]], [[segale-alignment]],
  [[2026-06-qwenasr-asr-regression-periodfix]].

## Sources
- code: `../codes/gigaspeech/convert_metricx_gigaspeech.py`, `filter_metricx_gigaspeech.py`, `../codes/metricx/`
