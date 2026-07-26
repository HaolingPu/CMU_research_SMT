---
title: Reference-selected best-of-5 (ref-free decode, ref-based selection)
type: experiment
tags: [consensus, selection, bleu, synthesis, ref-based-filtering]
sources:
  - ../codes/gigaspeech/future_sampling/select_bestof5_headroom.py
  - ../outputs/gigaspeech/consensus_decoding_prod/
created: 2026-07-19
updated: 2026-07-19
status: CLOSED NEGATIVE — bestof5 AND bestof4 both below top5-axis5; per-utt selection dead
---

# Reference-selected best-of-5 (ref-free decode, ref-based selection)

Successor to the closed anchor-and-veto line ([[2026-07-anchor-smoke500-sweep]]). Objective set
2026-07-19: close the **headline BLEU** gap to hibiki (COMET is already ~tied; BLEU gap is
largely surface-form — see [[comet-vs-bleu-ranking]], [[scoreboard]]). Constraint agreed with
Haoling: the **decoder stays reference-free; filtering/selection may use the reference**
(the pipeline already does — length_ratio_ref). Hibiki's BLEU edge comes from conditioning on
the ref; this design applies the same information at the only admissible stage: selection.

## Design

**Candidate pools** — five complete ref-free decodes of the SAME old-ASR 40k TSV
(`train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv`), all with segale-p24 + QE done:
`J_40k` (flagship axis5), `J_40k_fut100`, `J_40k_n100`, `J_40k_softvote`, `anchor_40k`
(anchor track). qwenasr pools excluded (different source set, known regression).

**Selection rule** — per utt_id, pick the pool whose prediction has max `metrics.bleu_char`
vs the frozen LLM reference, guarded by: candidate passes MetricX-QE ≤ 3.0 (its own pool's
scores) and length_ratio_ref ∈ [0.7, 1.5]. Winner's segale-aligned examples enter the union
manifest verbatim. Log per-pool win shares; anchor's share is the timing-risk watch item
(its burst style caused the trained-model degeneration — [[2026-07-anchor-smoke500-sweep]]).

**Step 0 — headroom gate (CPU-only, zero cost).** Oracle best-of-5 mean char-BLEU vs J_40k
alone, from the stored per-utt `metrics.bleu_char`. **Gate: oracle gain ≥ +2 char-BLEU**,
else pools are too similar for selection to matter → stop the direction at zero cost.
(The [[synthesis-bleu-not-predictive]] caveat is about ranking configs; a ~0 oracle means
selection cannot change the training targets at all — a stronger stop signal.)

**Downstream (unchanged flagship recipe)** — convert2swift SAMPLE_N=12500 seed 42 →
Megatron-SWIFT LoRA → HF export → infer ACL 6060 + Simul-tst-COMMON → eval vs top5-axis5
and hibiki.

**Success / kill** — trained BLEU ≥ +2 over top5-axis5 with COMET and LAAL held. Below that:
candidate diversity is the bottleneck → decide Approach B (widen pools with wording-diverse
decodes, re-select) or stop.

## Step-0 result (2026-07-19): gate PASSED, +10.9 oracle

`select_bestof5_headroom.py` over all 40,000 common utts (stored per-utt `bleu_char`):
flagship J_40k mean 59.21 → oracle best-of-5 (LR-guarded) 70.16, **gain +10.95**.
Pool means: J_40k 59.2 / fut100 59.6 / n100 59.8 / softvote 59.0 / anchor 66.5.
BUT anchor wins **55.6 %** of utts — the timing-risk watch item triggered. De-risk: built and
launched BOTH variants in parallel (training is only ~1 h):

| variant | manifest | n_sel | mean sel BLEU | anchor share | chain (conv→train→ACL/TST launchers) |
|---|---|---|---|---|---|
| bestof5refsel | `bestof5_refsel/manifest` | 26,025 | 68.04 | 46.4 % | 9365293→9365294→9365295/9365296 |
| bestof4refsel (no anchor) | `bestof4_noanchor_refsel/manifest` | 23,808 | 64.07 | 0 % | 9365289→9365290→9365291/9365292 |

No-anchor oracle (4 pools, 40k common): 65.03, gain **+5.82** — the safe variant keeps more
than half the headroom even before the union-of-QE-survivors effect.
Selection universe = union of QE(≤3.0) survivors per pool, LR guard 0.7–1.5; builder
`select_bestof5_build.py`. Both convert with SAMPLE_N=12500 seed 42; EXP names
`gigaspeech-zh-consensus-bestof{5,4}refsel-s-bsz4`. Interpretation guide: if bestof5 trains
clean AND wins, take it; if bestof5 degenerates (onomatopoeia/over-generation like anchor40k)
but bestof4 gains, the anchor pool is usable only via lower selection share or exclusion.

## Trained result — bestof5refsel (2026-07-19): NEGATIVE, +10.9 oracle → ~0 trained BLEU

Chain completed clean (conv 28 m, train 1 h, all infer/eval OK). vs top5-axis5:

| set | seg | BLEU Δ | COMET Δ | YAAL(CU) |
|---|---|---|---|---|
| ACL | 960 | 31.77 (−3.14) | .763 (−.024) | 1323 vs 1461 |
| ACL | 1920 | 37.57 (−2.04) | .798 (−.010) | 1923 vs 2176 |
| ACL | 2880 | 41.78 (**+1.66**) | .816 (+.004) | 2505 vs 2745 |
| ACL | 3840 | 40.00 (−0.14) | .816 (−.002) | 2959 vs 3107 |
| tst | 960 | 21.41 (−6.10) | .800 (−.031) | **18458 vs 3535** |
| tst | 1920 | 33.03 (+0.93) | .860 (+.002) | 1818 vs 1543 |
| tst | 2880 | 33.79 (−0.28) | .863 (−.004) | 2322 vs 2409 |
| tst | 3840 | 34.17 (−0.06) | .866 (−.006) | 2701 vs 2855 |

Reading: (1) **selection headroom does not transfer** — +10.9 synthesis-time oracle BLEU
becomes ≈0 trained BLEU at the useful latencies; extends [[synthesis-bleu-not-predictive]]
from config-ranking to per-utt selection. (2) **Anchor timing contamination is real but
latency-gated**: only tst seg960 blows up (YAAL 18.5 s vs 3.5 s, BLEU −6.1) — the burst/wait
style surfaces when the policy is forced to lowest latency; text stays coherent (COMET .80,
0 empty on ACL), so it is a timing pathology, not degeneration. (3) Hibiki gap unchanged
(ACL 3840: 40.0 vs 46.8). bestof4 (no anchor) will isolate whether (2) masked any gain;
per the interpretation guide, if bestof4 also shows no gain → per-utt surface-form
selection is dead and Approach B (pool diversity) is not worth running on the same logic.

bestof4 repair: ACL seg1 + tst seg4 hit vLLM engine-init flake → resubmitted twice (final
9368456_1/9368457_4), evals 9368432/9368465.

## Trained result — bestof4refsel no-anchor (2026-07-19): ALSO NEGATIVE → line closed

vs top5-axis5:

| set | seg | BLEU Δ | COMET Δ | YAAL(CU) |
|---|---|---|---|---|
| ACL | 960 | 33.71 (−1.20) | .775 (−.012) | 1295 |
| ACL | 1920 | 37.18 (−2.43) | .798 (−.010) | 1881 |
| ACL | 2880 | 38.60 (−1.52) | .809 (−.003) | 2454 |
| ACL | 3840 | 39.17 (−0.97) | .814 (−.003) | 2791 |
| tst | 960 | 24.53 (−2.98) | .818 (−.013) | 4446 |
| tst | 1920 | 26.28 (−5.82) | .825 (−.034) | **33213** |
| tst | 2880 | 32.64 (−1.43) | .859 (−.008) | 2211 |
| tst | 3840 | 33.28 (−0.95) | .864 (−.008) | 2597 |

Negative in EVERY cell — worse than bestof5 at most latencies despite excluding the anchor
pool. The tst seg1920 timing blow-up (YAAL 33 s) shows the mixture itself (per-utt style
switching between pools) destabilizes the policy even without anchor. Per the interpretation
guide: **per-utt surface-form selection is conclusively dead**; Approach B (pool widening)
not run, same logic applies. The lever training absorbs is systematic distribution shift,
not per-example picking — see [[2026-07-present-rank-winner]] for the successor family.

## Related
[[consensus-decoding]], [[future-sampling]], [[scoreboard]], [[synthesis-pipeline]],
[[2026-07-anchor-smoke500-sweep]], [[comet-vs-bleu-ranking]].
