---
title: Checkpoint Scoreboard (ACL 6060 dev)
type: comparison
tags: [eval, results, scoreboard]
sources:
  - ckpts/infinisst-omni/
created: 2026-06-01
updated: 2026-06-26
---

# Checkpoint Scoreboard (ACL 6060 dev)

Performance of trained [[infinisst-omni]] checkpoints on [[acl-6060]] dev, via
[[checkpoint-evaluation]]. Numbers are from each checkpoint's **latest** version's
`evaluation/acl_6060/<lang>/seg<N>/segmentation_output/scores.tsv` (omnisteval longform;
COMET = `Unbabel/XCOMET-XL`). **Latency = LongYAAL (CU)**, char-level ms (lower = faster).
Rank by **COMET** ([[comet-vs-bleu-ranking]]), not BLEU.

Source of truth: `ckpts/infinisst-omni/<exp>/<version>-hf/evaluation/`.

## Reading the board (key observations)
- **BLEU ≠ COMET.** [[east]] variants (`EAST-even`, `EAST-lowonly`) post the highest zh **BLEU**
  (43–47) but mediocre **COMET** (0.69–0.79) — surface match without semantic quality.
- **hibiki** (ref-based, sees the reference, cf. [[consensus-decoding]] note) tops both zh BLEU
  and COMET (0.82) — expected for a ref-based system.
- Among **ref-free** systems, the `consensus-top5-axis5` family ([[consensus-decoding]]) is the
  COMET leader (~0.817 @ seg3840) at much lower BLEU — it trades BLEU for COMET as designed.
  `top5-axis5` (5-axis, 20 futures) beats the futures=200 baseline `topk5` (0.806) at every
  latency, and soft-vote/100-future scaling don't help — see [[2026-06-consensus-axis5-vs-futures200]].
- **PA-40k** is a strong rule-based point (COMET ~0.81); **LA** ([[la-n-vs-wait-k]]) collapses at
  seg960 (burst mode: zh LA2 BLEU 5.6, latency 7200ms) — exclude seg960 for LA.
- **`top5-axis5` is trained on the OLD ASR** (`asr_filtered`) and is the canonical **old-asr+QE
  baseline**. Re-decoding with a **new Qwen-ASR (sentsplit)** regresses every latency by −4–6 BLEU /
  −0.05 COMET; period-fix recovers only ~+1 BLEU. The `FULL40k-win3*` and `top5-axis5-qwenasr*` rows
  below are the regressed new-asr runs — see [[2026-06-qwenasr-asr-regression-periodfix]].

## en→zh
| checkpoint | seg | BLEU | chrF | COMET | LongYAAL(CU) |
|---|---|---|---|---|---|
| consensus-top5-axis5 (OLD asr, baseline) | 1920 | 39.61 | 35.64 | **0.808** | 2176 |
| consensus-top5-axis5 (OLD asr, baseline) | 3840 | 40.14 | 35.77 | **0.817** | 3107 |
| consensus-FULL40k-win3 (NEW asr, +pfix) | 1920 | 35.47 | 33.50 | 0.761 | 1551 |
| consensus-FULL40k-win3 (NEW asr, +pfix) | 3840 | 34.22 | 32.50 | 0.765 | 2264 |
| consensus-FULL40k-win3-nopfix (NEW asr) | 3840 | 33.78 | 32.47 | 0.763 | 2159 |
| consensus-top5-axis5-qwenasr-fixed (NEW asr) | 1920 | 36.12 | 32.68 | 0.777 | 1660 |
| consensus-top5-axis5-qwenasr (NEW asr) | 3840 | 32.69 | 31.43 | 0.755 | 2130 |
| consensus-top5-axis5-fut100_n100 | 3840 | 39.30 | 36.01 | 0.816 | 2757 |
| consensus-top5-axis5-sv | 3840 | 38.65 | 34.78 | 0.812 | 2894 |
| consensus-topk5_v2 | 2880 | 38.03 | 34.46 | 0.814 | 2519 |
| consensus-topk5_k4 | 3840 | 39.33 | 35.89 | 0.814 | 2842 |
| consensus-topk5 | 3840 | 37.42 | 33.64 | 0.806 | 2855 |
| hibiki (ref-based) | 3840 | 46.76 | 40.54 | **0.820** | 3326 |
| PA-40k | 1920 | 40.62 | 35.94 | 0.809 | 1954 |
| LA-40k-seg14-LA2 | 3840 | 28.40 | 30.80 | 0.780 | 2772 |
| EAST-even | 3840 | 46.83 | 39.36 | 0.789 | 3533 |
| EAST-lowonly | 3840 | 42.13 | 35.53 | 0.728 | 2998 |
| Simul-MuST-C-fixed-v2 | 3840 | 45.85 | 38.77 | 0.763 | 3184 |

_(Per-checkpoint full seg960/1920/2880/3840 rows live in each `…-hf/evaluation/` dir; the table
above shows headline seg points. LA-40k-s and LA-40k-seg13 have no eval output at the standard
path — run pending / nested differently.)_

## en→ja
| checkpoint | seg | BLEU | chrF | COMET | LongYAAL(CU) |
|---|---|---|---|---|---|
| EAST-latency2mult | 2880 | 27.92 | 39.87 | 0.708 | 2875 |
| EAST-latency2mult | 3840 | 28.43 | 40.33 | **0.743** | 3561 |
| EAST-low | 3840 | 23.17 | 36.91 | 0.645 | 3059 |
| Simul-MuST-C | 3840 | 23.84 | 35.24 | 0.591 | 3554 |

## en→de
| checkpoint | seg | BLEU | chrF | COMET | LongYAAL(CU) |
|---|---|---|---|---|---|
| EAST-latency2mult | 2880 | 33.96 | 66.36 | **0.909** | 4720 |
| EAST-latency2mult | 3840 | 34.07 | 65.55 | 0.903 | 5595 |
| EAST-low | 3840 | 25.61 | 63.57 | 0.888 | 3046 |
| Simul-MuST-C | 3840 | 27.97 | 64.87 | 0.885 | 3293 |

## Related
- [[checkpoint-evaluation]], [[latency-quality-tradeoff]], [[comet-vs-bleu-ranking]],
  [[infinisst-omni]], [[consensus-decoding]], [[la-n-vs-wait-k]], [[east]], [[acl-6060]],
  [[2026-07-anchor-smoke500-sweep]] (anchor_40k → train → new scoreboard row pending).

## Sources
- results: `ckpts/infinisst-omni/<exp>/<version>-hf/evaluation/acl_6060/<lang>/seg<N>/segmentation_output/scores.tsv`
