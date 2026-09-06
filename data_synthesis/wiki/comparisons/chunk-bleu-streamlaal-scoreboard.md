---
title: Chunk-level BLEU / StreamLAAL scoreboard (ACL 6060 dev, en→zh, 12.5K training)
type: comparison
tags: [eval, results, scoreboard, chunk-bleu, streamlaal]
sources:
  - user-provided results table (Haoling, 2026-09-05; pre-2026-06 training runs, 12.5K instances each)
  - ckpts/infinisst-omni/gigaspeech-zh-consensus-ambiguity-q38-gemma-q36-fsetv2-prefixnorm-strict-40k-r1-20260831-s-bsz4/v0-20260906-013823-hf/evaluation/*/en-zh/seg*/scores.tsv
created: 2026-09-05
updated: 2026-09-05
---

# Chunk-level BLEU / StreamLAAL scoreboard (ACL 6060 dev, en→zh)

This is the **second metric family** used in the project, distinct from the longform numbers in
[[scoreboard]]:

| metric | what it is | where it comes from |
|---|---|---|
| **StreamLAAL-BLEU** ("chunk-level BLEU") | sacreBLEU over the streaming hypothesis as SimulEval emits it, `zh` tokenizer, no re-segmentation | `<ckpt>/evaluation/<set>/en-zh/seg<N>/scores.tsv` (SimulEval) |
| **StreamLAAL (ms)** | length-adaptive average lagging of the stream | older streamLAAL script (now disabled in `infer_slurm.sh`); numerically ≈ **LongYAAL (CU)** from [[checkpoint-evaluation]] (hibiki @3840: 3319 vs 3326) |
| longform BLEU / XCOMET | hypothesis re-aligned to reference sentences with mwerSegmenter, then scored | `segmentation_output/scores.tsv` (omnisteval) — the paper's primary quality numbers |

Chunk-level BLEU runs ~5 points higher than longform BLEU for the same system (hibiki 51.85 vs
46.76 @3840) because it never penalises sentence-boundary drift. Rank by COMET
([[comet-vs-bleu-ranking]]); use this table for continuity with the pre-June experiment log.
All rows below: **ACL 6060 dev**, en→zh, Qwen3-Omni-30B LoRA, **12.5K training instances** unless
noted. SimulEval's own LAAL/AL columns are unreliable on long-form audio (negative values) and are
not reported; latency is StreamLAAL / LongYAAL(CU).

## Reference-free synthesis baselines

| system | chunk ms | StreamLAAL ms | chunk BLEU |
|---|---|---|---|
| EAST | 960 | 1184 | 36.17 |
| EAST-low-mult | 960 | 1248 | 39.27 |
| EAST-low-mult-fixed | 960 / 1920 / 2880 / 3840 | 1323 / 1989 / 2363 / 2769 | 40.10 / 41.96 / 42.75 / 43.54 |
| EAST-latency2mult | 960 / 1920 / 2880 / 3840 | 1264 / 1978 / 2767 / 3599 | 39.74 / 45.74 / 47.52 / 49.94 |
| Refined-EAST | 960 | 1269 | 36.42 |
| Refined-EAST-low-mult | 960 | 1227 | 38.50 |
| Refined-EAST-low-mult-fixed | 960 / 1920 / 2880 / 3840 | 1176 / 1845 / 2233 / 2740 | 39.75 / 43.04 / 42.82 / 43.20 |
| Refined-EAST-latency2mult | 960 / 1920 / 2880 / 3840 | 1247 / 1962 / 2864 / 3541 | 40.20 / 45.31 / 48.17 / 49.59 |
| Simul-MuST-C | 960 | 1664 | 32.55 |
| Simul-MuST-C-fixed | 960 / 1920 / 2880 / 3840 | 1165 / 2128 / 2886 / 3259 | 38.47 / 43.94 / 45.23 / 45.44 |
| Simul-MuST-C-fixed_v2 | 960 / 1920 / 2880 / 3840 | 1125 / 2006 / 2607 / 3057 | 38.32 / 46.34 / 48.62 / 48.26 |

## Reference-based baselines (see the reference at synthesis time)

| system | chunk ms | StreamLAAL ms | chunk BLEU |
|---|---|---|---|
| Word-Alignment (InfiniSST) | 960 / 1920 / 2880 / 3840 | 1180 / 1808 / 2251 / 2615 | 42.33 / 45.60 / 48.19 / 48.48 |
| Hibiki | 960 / 1920 / 2880 / 3840 | 1506 / 2210 / 2828 / 3319 | 46.08 / 49.17 / 51.17 / **51.85** |

## Consensus family ([[consensus-decoding]])

| system | chunk ms | StreamLAAL / LongYAAL(CU) ms | chunk BLEU |
|---|---|---|---|
| Consensus top-k=5 (futures=200 baseline, `topk5`) | 960 / 1920 | 1310 / 1891 | 36.13 / 40.65 |
| Consensus top-k=10 | 960 / 1920 | 1171 / 1755 | 33.03 / 37.73 |
| **Ambiguity future set, 2026-09** ([[2026-09-ambiguity-fsetv2-40k]]; 17,306 instances) | 960 / 1920 / 2880 / 3840 | 1345 / 1970 / 2566 / 3031 (LongYAAL CU) | **46.45 / 54.76 / 56.56 / 57.23** |

Seg2880/3840 chunk BLEU for the two older consensus rows were never recorded. The `top5-axis5`
flagship's chunk-level BLEU is not in this table either; its longform numbers are in [[scoreboard]].

## Simul-tst-COMMON, chunk-level, ambiguity run only

| chunk ms | LongYAAL(CU) ms | chunk BLEU |
|---|---|---|
| 960† / 1920 / 2880 / 3840 | 15495† / 2078 / 2442 / 2884 | 27.99† / 54.71 / 57.04 / 58.14 |

† degenerate at seg960 (see [[2026-09-ambiguity-fsetv2-40k]]).

## Reading
- On chunk-level BLEU the 2026-09 ambiguity run is **+5.4 over Hibiki at 3840** (57.2 vs 51.9)
  and +7 over the best EAST variant, at lower latency than either. On longform BLEU the margin
  over Hibiki is +1.0 and XCOMET is −0.02 — see [[scoreboard]] and [[comet-vs-bleu-ranking]]
  before drawing conclusions. The ambiguity run also trained on 17,306 instances vs 12.5K here.

## Related
- [[scoreboard]], [[checkpoint-evaluation]], [[comet-vs-bleu-ranking]], [[consensus-decoding]],
  [[east]], [[salami]], [[infinisst-omni]], [[acl-6060]], [[2026-09-ambiguity-fsetv2-40k]],
  [[2026-06-consensus-axis5-vs-futures200]].
