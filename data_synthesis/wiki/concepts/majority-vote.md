---
title: Majority Vote
type: concept
tags: [synthesis, ensemble]
sources:
  - ../codes/gigaspeech/future_sampling/llm_future_sampling_majority_vote.py
created: 2026-06-01
updated: 2026-06-07
---

# Majority Vote

Simple ensemble selection inside [[future-sampling]]: sample multiple LLM outputs per chunk and
commit the most common translation. A baseline alternative to [[consensus-decoding]]'s
distribution-level agreement and to the [[thinking-policy]] judge.

Relaxing consensus's hard token-intersection into a majority vote (the `soft-vote` variant,
`--min-voters-ratio 0.75`) was tried and **lost** to the hard 5-axis consensus at trained eval —
see [[2026-06-consensus-axis5-vs-futures200]]. Looser commitment does not help.

Files: `../codes/gigaspeech/future_sampling/llm_future_sampling_majority_vote.py` and a `_v2`.

## Related
- [[future-sampling]], [[consensus-decoding]], [[2026-06-consensus-axis5-vs-futures200]].

## Sources
- code: `../codes/gigaspeech/future_sampling/llm_future_sampling_majority_vote*.py`
