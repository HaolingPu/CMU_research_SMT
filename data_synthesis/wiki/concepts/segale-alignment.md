---
title: SeGALE Alignment
type: concept
tags: [synthesis, alignment]
sources:
  - ../codes/gigaspeech/future_sampling/scripts/segale/prepare_segale_shards.py
  - ../codes/gigaspeech/future_sampling/scripts/segale/merge_aligned_shards.py
created: 2026-06-01
updated: 2026-06-26
---

# SeGALE Alignment

Token-level alignment module that matches source chunks to target segments, used for the
post-check truncation step in [[future-sampling]] (truncating an over-translation to the
observed-safe prefix). Sharded for multi-GPU runs.

Files: `../codes/gigaspeech/future_sampling/scripts/segale/prepare_segale_shards.py` (shard prep),
`merge_aligned_shards.py` (merge), plus 8/24-GPU run scripts. Related word-level aligner:
awesome-align / simalign (see [[future-sampling]]).

It is also stage 2 of the canonical post-decode QE pipeline (`submit_J40k_post.sh`): align →
per-sentence [[metricx]] QE-MAX≤3 → length filter, the filtering used in
[[2026-06-qwenasr-asr-regression-periodfix]] (which found QE filtering rescues a collapsed new-asr
seg960 from BLEU 6.9 → ~30, but cannot close the old↔new ASR gap).

## Related
- [[future-sampling]], [[mfa]] (audio-text forced alignment, a different alignment role),
  [[metricx]], [[2026-06-qwenasr-asr-regression-periodfix]].

## Sources
- code: `../codes/gigaspeech/future_sampling/scripts/segale/`
