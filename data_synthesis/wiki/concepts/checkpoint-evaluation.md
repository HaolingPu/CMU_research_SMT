---
title: Checkpoint Evaluation
type: concept
tags: [eval, pipeline]
sources:
  - scripts/infer/eval_all_ckpts.sh
  - scripts/infer/normalize_instances.py
created: 2026-06-01
updated: 2026-06-01
---

# Checkpoint Evaluation

Post-training assessment pipeline. For each checkpoint (listed in `scripts/infer/ckpts*.txt`) ×
4 segment sizes: run [[streaming-inference]] → `instances.log` → NFKC-normalize with
`normalize_instances.py` → **omnisteval longform** → multi-metric `scores.tsv`.

Metrics: BLEU (SacreBLEU, char-level for zh/ja), COMET (`Unbabel/XCOMET-XL`), chrF, plus latency
(LongYAAL). Test set: **ACL 6060 dev** ([[acl-6060]]), 4 reference languages. Language-specific
tokenizers (MOSES; ja-mecab). Driver: `scripts/infer/eval_all_ckpts{,_ja,_de,_v2}.sh`.

Results land in each checkpoint's `…-hf/evaluation/acl_6060/<lang>/seg<N>/` dir (the
`segmentation_output/scores.tsv` is the COMET-bearing one), are consolidated in the [[scoreboard]],
and feed the [[latency-quality-tradeoff]] plots.

## Related
- [[streaming-inference]], [[latency-quality-tradeoff]], [[acl-6060]], [[comet-vs-bleu-ranking]], [[infinisst-omni]], [[babel-cluster]].

## Sources
- code: `scripts/infer/eval_all_ckpts.sh`, `normalize_instances.py`, `ckpts*.txt`
