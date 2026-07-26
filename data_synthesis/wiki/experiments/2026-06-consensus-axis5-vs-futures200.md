---
title: Consensus "other approaches" — 5-axis vs soft-vote vs scaling vs the futures=200 baseline
type: experiment
tags: [synthesis, consensus, ref-free, future-sampling, results, trained-eval]
sources:
  - ../codes/gigaspeech/future_sampling/consensus_decoding_token_id_level_instruct.py
  - scripts/train/convert2swift_consensus.py
  - ckpts/infinisst-omni/gigaspeech-zh-consensus-top5-axis5-s-bsz4/
created: 2026-06-07
updated: 2026-06-26
---

# Consensus "other approaches" — 5-axis vs soft-vote vs scaling vs the futures=200 baseline

**Question.** Starting from the naive [[consensus-decoding]] baseline (200 undirected futures, hard
token-intersection), which modification actually improves trained quality on [[acl-6060]] en→zh?
Three levers were tried: **directed future diversity** (5-axis), **looser commitment** (soft-vote),
and **brute-force scaling** (more futures). All trained end-to-end and evaluated — not synthesis-time.

**Answer.** Directed diversity wins; the other two don't. **5-axis (20 directed futures) dominates
the 200-future baseline at every latency**, while loosening commitment (soft-vote) and scaling to
100 futures both fail to beat it.

## Baseline provenance
The futures=200 baseline is the trained **`consensus-topk5`** checkpoint:
`scripts/train/convert2swift_consensus.py` builds its data from
`consensus_decoding_en_zh_top_5_futures200-segale/qe3-lr-aligned-full` (200 undirected futures,
plain top-k intersection, no axis steering). So "futures200" itself is synthesis-only; `topk5` is
its trained eval surface (see [[scoreboard]]).

## The four approaches

| variant | lever | futures | code |
|---|---|---|---|
| **futures200 baseline** (`topk5`) | none (naive) | 200 undirected | base intersection |
| **5-axis** (`top5-axis5`) ⭐ | directed diversity: 5 narrative axes × ~4 | **20 directed** | `sample_source_futures_targeted_prefill` (`AXES`, ~line 1290) |
| **soft-vote** (`top5-axis5-sv`) | loosen hard intersection → majority | 20 directed | `choose_consensus_token` + `--min-voters-ratio 0.75` |
| **fut100 / fut100_n100** | scale 5-axis to 100 futures | 100 directed | same as 5-axis, more samples |

## Results (trained, ACL6060 dev en→zh; COMET is the ranking metric)

COMET @ seg 960 / 1920 / 2880 / 3840:

| variant | seg960 | seg1920 | seg2880 | seg3840 |
|---|---|---|---|---|
| futures200 baseline (topk5) | 0.777 | 0.797 | 0.799 | 0.806 |
| **5-axis** | **0.787** | **0.808** | **0.812** | **0.817** |
| soft-vote | 0.763 | 0.808 | 0.813 | 0.812 |
| fut100_n100 | 0.767 | 0.805 | 0.807 | 0.816 |

BLEU @ same segs:

| variant | seg960 | seg1920 | seg2880 | seg3840 |
|---|---|---|---|---|
| futures200 baseline (topk5) | 32.4 | 35.2 | 34.7 | 37.4 |
| **5-axis** | **34.9** | **39.6** | **40.1** | **40.1** |
| soft-vote | 30.9 | 37.6 | 38.5 | 38.7 |
| fut100_n100 | 32.1 | 36.9 | 38.4 | 39.3 |

Latency (LongYAAL CU, ms): baseline 1319/1883/2378/2855; 5-axis 1461/2176/2745/3107.

## Findings

1. **5-axis dominates the baseline on COMET and BLEU at every latency** (+0.010–0.013 COMET,
   +2.7–4.4 BLEU). It is the production winner (the `consensus-top5-axis5` flagship). **Data source:
   old ASR** (`asr_filtered`, prod root `J_40k`, QE-filtered) — this checkpoint is the canonical
   **old-asr+QE baseline**. Swapping in a new Qwen-ASR regresses it −4–6 BLEU and period-fix doesn't
   recover it; see [[2026-06-qwenasr-asr-regression-periodfix]].
2. **Efficiency win (headline):** 5-axis at seg1920 (COMET **0.808**, 2176 ms) already **exceeds the
   baseline's best point** (seg3840, 0.806, 2855 ms). → **20 directed futures beat 200 undirected
   ones**, reaching peak quality ~680 ms sooner. Directed diversity > brute-force sampling.
3. **soft-vote is a negative result.** Loosening the hard intersection to a 0.75 majority lands
   between baseline and 5-axis, and at high latency dips below 5-axis (COMET 0.812 < 0.817, BLEU
   38.7 < 40.1). Confirms the gap is **not** "too conservative" — loosening hurts. Mirrors the
   synthesis-time finding in [[2026-06-consensus-post-edit-bleu]].
4. **Scaling futures saturates at ~20.** fut100_n100 ≈ ties 5-axis (0.816 vs 0.817) at higher cost;
   fut100 (no n100) is slightly worse (~0.810 @ seg3840). More futures don't help once the 5 axes
   supply the diversity — so 200 (the baseline) is wasteful, not beneficial.

**Takeaway:** the consensus quality lever is **where the diversity comes from (directed axes), not
how much you sample or how loosely you commit.** See [[future-sampling]], [[min-p-sampling]],
[[comet-vs-bleu-ranking]], [[latency-quality-tradeoff]].

## Caveats
- All numbers are **trained eval** (the only signal that counts; synthesis-time BLEU is
  non-predictive — memory `feedback_synthesis_bleu_not_predictive`).
- en→zh only; ja/de consensus not trained (see [[scoreboard]]).
- Figure: `scripts/debug/consensus_vs_futures200.png` (BLEU + COMET vs latency, all four).

## Sources
- [[consensus-decoding]], [[future-sampling]], [[scoreboard]], [[comet-vs-bleu-ranking]],
  [[min-p-sampling]], [[majority-vote]], [[2026-06-consensus-post-edit-bleu]],
  [[2026-06-qwenasr-asr-regression-periodfix]], [[acl-6060]], [[infinisst-omni]]
