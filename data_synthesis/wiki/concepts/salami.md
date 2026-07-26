---
title: SALAMI
type: concept
tags: [synthesis, segmentation]
sources:
  - ../codes/gigaspeech/salami/llm_output_salami.py
  - ../codes/gigaspeech/salami/pipeline.sh
created: 2026-06-01
updated: 2026-06-01
---

# SALAMI

Synthesis method that emits a **segmented-pairs** format — a list of `[source, target]` pairs —
instead of [[east]]'s per-latency arrays. Produced by `../codes/gigaspeech/salami/llm_output_salami.py`
(8-GPU array), then run through the shared [[synthesis-pipeline]] with SALAMI-specific stages:

- `fix_llm_raw.py --sync_zh_punct` — restore punctuation, sync Chinese punctuation across boundaries.
- `map_salami_to_offline_gigaspeech.py` — convert segmented pairs to the offline trajectory layout.
- `find_bad_json_gigaspeech.py --allow-one-word` — relaxed [[mfa]] alignment check.

Output: `../outputs/gigaspeech/train_xl_salami/`.

## Related
- [[east]], [[synthesis-pipeline]], [[gigaspeech]].

## Sources
- code: `../codes/gigaspeech/salami/`
