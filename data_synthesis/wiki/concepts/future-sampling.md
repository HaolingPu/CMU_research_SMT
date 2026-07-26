---
title: Future Sampling
type: concept
tags: [synthesis, policy]
sources:
  - ../codes/gigaspeech/future_sampling/llm_future_sampling_final.py
  - ../codes/gigaspeech/future_sampling/llm_future_sampling_core.py
  - ../codes/gigaspeech/future_sampling/LLM_FUTURE_SAMPLING_FINAL_WALKTHROUGH.md
created: 2026-06-01
updated: 2026-06-07
---

# Future Sampling

Online simultaneous-translation synthesis policy for [[gigaspeech]] (the successor to [[east]]'s
offline segmentation). At each ~960ms chunk: a **base** LLM samples several plausible English
continuations; an **instruct** LLM translates each; an aligner truncates to the observed-safe
prefix ([[segale-alignment]] / awesome-align); a judge scores candidates and a READ/WRITE
decision is made from consensus + direction consistency. Dual-model serving via [[vllm]] (local
base + served instruct). The paid GPT/DeepSeek sampler has a local replacement:
[[qwen35-122b-sampler]] (thinking mode; see that page for cost/scaling data).

Canonical impl: `../codes/gigaspeech/future_sampling/llm_future_sampling_final.py` (+ walkthrough
MD). Earlier: `llm_future_sampling_core.py`, `_v2/_v3`.

## Variants
- **b1 / b2** — longest-common-prefix (LCP) code-based consensus instead of an LLM judge;
  `_b1.py`, `_b2.py` (70% confidence threshold).
- [[majority-vote]] — commit the most common candidate.
- [[thinking-policy]] — a reasoning model decides READ/WRITE.
- [[consensus-decoding]] — distribution-level agreement across two base models.
- Diversity of the candidate set is swept in [[min-p-sampling]].

The commit rule sits at a conservative sweet spot: loosening it (soft-vote majority) or
post-editing its output to chase BLEU both regress — see [[2026-06-consensus-post-edit-bleu]].
The lever that *does* help is directed candidate diversity (5-axis), which beats both the naive
futures=200 baseline and brute-force scaling — see [[2026-06-consensus-axis5-vs-futures200]].

Output: `../outputs/gigaspeech/train_xl_future_sampling_final/`, processed by [[synthesis-pipeline]]
+ [[metricx]] filtering.

In [[2026-07-anchor-smoke500-sweep]] futures switch role from choosing wording to only vetoing
commit timing (anchor-and-veto): the anchor proposes greedy text, futures decide how far to commit.

## Sources
- code: `../codes/gigaspeech/future_sampling/`
