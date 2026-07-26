---
title: Latency–Quality Tradeoff
type: concept
tags: [eval, latency, analysis]
sources:
  - scripts/debug/plot_latency_quality_3models.py
  - scripts/infer/plot_latency_quality_zh.py
created: 2026-06-01
updated: 2026-06-07
---

# Latency–Quality Tradeoff

The core evaluation lens: quality (BLEU / COMET / chrF / BLEURT / TASER) plotted against latency
(**LongYAAL**, character-level ms; legacy StreamLAAL) across the 4 segment sizes from
[[streaming-inference]]. Each policy traces a Pareto frontier; [[east]]-style strategies aim to
dominate [[la-n-vs-wait-k]] / Simul-MuST-C baselines.

Plot scripts: `scripts/debug/plot_latency_quality_3models.py` (LongYAAL vs BLEU/COMET/chrF over
consensus/PA/LA models; skips seg960 for LA where it degenerates to burst mode),
`plot_latency_quality_future_aware.py`, and per-language `scripts/infer/plot_latency_quality_{zh,ja,de}.py`.

Note: rank policies by **COMET**, not BLEU — see [[comet-vs-bleu-ranking]].

A worked example of reading the frontier "left": 5-axis consensus reaches the futures=200
baseline's *best* COMET ~680 ms sooner — see [[2026-06-consensus-axis5-vs-futures200]]
(`scripts/debug/consensus_vs_futures200.png`).

## Related
- [[checkpoint-evaluation]], [[streaming-inference]], [[comet-vs-bleu-ranking]],
  [[2026-06-consensus-axis5-vs-futures200]], [[2026-07-anchor-smoke500-sweep]] (anchor gate
  strictness trades LAAL for target quality: +1.8 LAAL at +7 char-BLEU).

## Sources
- code: `scripts/debug/plot_latency_quality*.py`, `scripts/infer/plot_latency_quality_*.py`
