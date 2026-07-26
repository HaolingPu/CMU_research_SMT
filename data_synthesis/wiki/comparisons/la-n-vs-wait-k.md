---
title: LA-N vs Wait-k
type: comparison
tags: [synthesis, policy, baseline]
sources:
  - ../codes/gigaspeech/rule-based-SMT/
  - scripts/train/convert2swift_LA.py
created: 2026-06-01
updated: 2026-06-01
---

# LA-N vs Wait-k

Rule-based simultaneous-MT policies under `../codes/gigaspeech/rule-based-SMT/` — **local-agreement
(LA-N)**, **wait-k**, and **prefix-alignment (PA)**. They are **offline synthesis policies** (with
quality eval), not merely baselines, and each has a trained counterpart (`LA_s`, `PA_s`) via
[[dataset-conversion-pipeline]] → [[megatron-swift]] → [[infinisst-omni]].

LA conversion uses random chunk-size multipliers (1–12×) to simulate latency regimes. On the
latency axis, LA degenerates to burst mode at seg960 (skipped in [[latency-quality-tradeoff]]
plots).

## Gotchas
- LA-N's weak eval is a **data/policy** issue, not infra: zh runs use bundled Megatron, train
  loss (~1.03) is non-predictive ([[megatron-swift]]).
- Synthesis-time BLEU does not predict trained-model eval BLEU — rank by COMET ([[comet-vs-bleu-ranking]]).

## Related
- [[east]], [[consensus-decoding]], [[latency-quality-tradeoff]].

## Sources
- code: `../codes/gigaspeech/rule-based-SMT/`, `scripts/train/convert2swift_{LA,PA}.py`
