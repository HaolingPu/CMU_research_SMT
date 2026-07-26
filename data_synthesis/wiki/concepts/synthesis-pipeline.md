---
title: Synthesis Pipeline (EAST / SALAMI / Future)
type: concept
tags: [synthesis, pipeline]
sources:
  - ../codes/gigaspeech/east/pipeline.sh
  - ../codes/gigaspeech/salami/pipeline.sh
created: 2026-06-01
updated: 2026-06-01
---

# Synthesis Pipeline

The end-to-end offline data-synthesis pipeline that turns a [[gigaspeech]] manifest into
streaming simultaneous-translation training data. Shared shape across [[east]], [[salami]], and
the future-sampling methods; orchestrated by `pipeline.sh` per method.

## Stages

1. **LLM segmentation** (8-GPU array, vllm env) — segment full source into chunked source/target.
   `../codes/gigaspeech/llm_output_gigaspeech_trajectory.py` (EAST), `salami/llm_output_salami.py`.
2. **Pipeline processing** (1 GPU) — `fix_llm_raw.py` → `post_process_llm_output_gigaspeech.py`
   (language-aware joining: space for de/en, none for zh/ja) → `find_bad_json_gigaspeech.py`
   ([[mfa]] alignment check) → `multi_trajectory_gigaspeech.py` (build ~960ms streaming chunks) →
   `convert_metricx_gigaspeech.py` (→ [[metricx]] input, split into 8 shards).
3. **MetricX QE predict** (8-GPU array) — score each segment, see [[metricx]].
4. **Assembly & filter** — `filter_metricx_gigaspeech.py` (keep prediction ≤ 3.0) →
   `final_output_gigaspeech.py` (per-`recording_id`/`latency` JSONL).

The committed token chunks rest on a ~960ms grid (15360 samples @ 16kHz); see
[[dataset-conversion-pipeline]] for how these become training instances.

## Method differences
- [[east]]: English-only, no `fix_llm_raw`.
- [[salami]]: adds `fix_llm_raw` (`--sync_zh_punct`) + salami→offline mapping + `--allow-one-word`.
- Future methods ([[future-sampling]], [[consensus-decoding]]) replace stage 1's segmentation
  with online READ/WRITE decisions; current prod variant: anchor-and-veto
  ([[2026-07-anchor-smoke500-sweep]], anchor_40k decode).

## Sources
- code: `../codes/gigaspeech/east/pipeline.sh`, `../codes/gigaspeech/salami/pipeline.sh`
- outputs land under `../outputs/gigaspeech/train_xl_*`
- every stage is an sbatch array on [[babel-cluster]] (8/24-GPU shards; mind partition MaxTime + preempt eviction)
