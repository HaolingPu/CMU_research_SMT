---
title: GigaSpeech
type: entity
tags: [dataset, speech]
sources:
  - ../codes/gigaspeech/
created: 2026-06-01
updated: 2026-06-26
---

# GigaSpeech

Primary speech corpus driving the synthesis pipeline (English source; targets zh/de/ja). The XL
manifest `train_xl_case_robust_asr-filtered.tsv` provides `src_text_full` and `src_trajectory`
(960ms chunks), the backbone for every method here.

Feeds the [[synthesis-pipeline]] across methods: [[east]], [[salami]], [[future-sampling]],
[[consensus-decoding]]. Synthesized data is then converted for training via
[[dataset-conversion-pipeline]] and trained into [[infinisst-omni]] checkpoints.

## ASR source variants (eval_datasets/ TSVs)
The source transcript can come from different ASR systems — this matters because **ASR quality
dominates trained eval** (see [[2026-06-qwenasr-asr-regression-periodfix]]):
- **OLD asr** = `train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv` (original non-Qwen ASR).
  The flagship `consensus-top5-axis5` baseline trains on this.
- **NEW asr** = `train_xl_case_robust_qwenasr_filtered…` and `…_qwenasr_sentsplit…` (Qwen-ASR; the
  `sentsplit` variant adds spaCy sub-sentence splitting + a chunk-start 。 artifact). Both regress
  −4–6 BLEU vs old asr.
- `train_xl_INTERSECT_old-asr.tsv` / `…_qwenasr-sentsplit.tsv` are the common-ID intersection sets
  used for the no-QE control runs.

## Related
- [[synthesis-pipeline]], [[metricx]], [[mfa]], [[acl-6060]] (the separate eval set),
  [[2026-06-qwenasr-asr-regression-periodfix]].

## Sources
- code: `../codes/gigaspeech/`; manifests under `li_lab/siqiouya/datasets/gigaspeech/`
