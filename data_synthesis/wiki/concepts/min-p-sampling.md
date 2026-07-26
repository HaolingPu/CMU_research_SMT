---
title: Min-p / Top-k Sampling
type: concept
tags: [synthesis, sampling, ablation]
sources:
  - ../codes/gigaspeech/future_sampling/scripts/minp/
  - ../codes/gigaspeech/future_sampling/scripts/topk/
created: 2026-06-01
updated: 2026-06-07
---

# Min-p / Top-k Sampling

Diversity-control ablations for the base model's future candidate set in [[future-sampling]] and
[[consensus-decoding]]. Min-p sweeps thresholds (1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.3); a parallel
top-k family exists. The candidate set feeds the consensus / min-p selection of the committed
token.

What matters for trained quality is *where* the diversity comes from, not raw sample count:
directed 5-axis sampling with 20 futures beats 200 undirected futures, and scaling 5-axis to 100
futures adds nothing — see [[2026-06-consensus-axis5-vs-futures200]].

Files: `../codes/gigaspeech/future_sampling/scripts/minp/` (sweep infra, `run_minp_common.sh`,
MetricX QE sbatches), `scripts/topk/`.

## Related
- [[future-sampling]], [[consensus-decoding]], [[metricx]], [[2026-06-consensus-axis5-vs-futures200]].

## Sources
- code: `../codes/gigaspeech/future_sampling/scripts/minp/`, `scripts/topk/`
