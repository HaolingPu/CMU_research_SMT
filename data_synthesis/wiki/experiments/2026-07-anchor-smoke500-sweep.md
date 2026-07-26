---
title: Anchor-and-veto smoke sweep (500 utts, A/B/C/D) — strict gate wins
type: experiment
tags: [anchor, veto, consensus, smoke, register, synthesis]
sources:
  - ../codes/gigaspeech/future_sampling/consensus_decoding_anchor.py
  - ../codes/gigaspeech/future_sampling/run_anchor_smoke500.sbatch
  - ../codes/gigaspeech/future_sampling/analyze_anchor_smoke.py
  - ../outputs/gigaspeech/consensus_decoding_prod/anchor_smoke500/
created: 2026-07-12
updated: 2026-07-12
---

# Anchor-and-veto smoke sweep (500 utts, A/B/C/D) — strict gate wins

Executes the validation protocol of [[2026-07-consensus-register-forensics]] fix plan #1
(propose-from-present, futures-verify). Four dual-track smoke runs (SLURM 9229259/9229317/
9229318/9230169), each 500 utts of the **old-ASR** frozen-reference TSV (the flagship
top5-axis5 data), anchor and vanilla consensus decoded on the SAME futures. Paired analysis
via `analyze_anchor_smoke.py`.

## Results (mean over 500; paired anchor − consensus)

| variant | veto min-p / top-k / voters | anchor char-BLEU | consensus | Δ paired (W/L) | LAAL a/c | 4-gram recall a/c |
|---|---|---|---|---|---|---|
| **A** | 0.05 / 5 / **1.00** | **67.25** | 60.27 | **+6.98** (347/122) | 6.55 / 4.75 | **0.578** / 0.494 |
| D | 0.05 / 5 / 0.95 | 64.47 | 59.98 | +4.49 (312/148) | 5.74 / 4.68 | 0.545 / 0.489 |
| B | 0.02 / 10 / 0.90 | 55.40 | 60.22 | −4.83 (139/316) | 4.49 / 4.68 | 0.441 / 0.490 |
| C | 0.01 / 20 / 0.80 | 45.49 | 60.40 | −14.92 (47/441) | 3.66 / 4.67 | 0.334 / 0.493 |

## Findings

- **Anchor helps, but only with the strict gate.** A hits the forensics success criteria:
  char-BLEU vs frozen ref 60→67 (target 65+), 4-gram recall 0.49→0.58 (target 0.6, near-miss).
  Quality is **monotone in gate strictness**; loosening (B/C) loses quality far faster than it
  buys latency. A = exactly the config fix plan #1 prescribed ("keep the flagship's strictness").
- **Register normalized** (the mechanism the method targets): A's marker profile tracks the
  frozen ref where consensus diverged — 因此 7.2 vs cons 10.0 (ref 5.8), 将 35.2 vs 62.7
  (ref 32.3), 把 63.1 vs 39.3 (ref 66.4). Anchor never inherits the soft-vote formal-register
  catalog.
- **Latency caveat**: A commits later — LAAL 6.55 vs consensus 4.75 (+1.8); D +1.06. Per the
  [[latency-quality-tradeoff]] concern in the forensics protocol this confounds the smoke
  leading indicators; final judgment deferred to trained-model eval (BLEU/COMET *and* LAAL).
- Veto behavior at A: 59 % of chunks WRITE, mean commit 4.7 tokens when >0; top vetoed tokens
  are pronouns/sentence-enders (他, 。, 那) — i.e. the veto mostly blocks premature commitment,
  as designed.

## Decision (2026-07-12)

Anchor judged helpful → launched **anchor_40k** production decode with variant A params on the
old-ASR TSV (40k rows, 16 tasks × 2500): SLURM arrays 9239309 (preempt 0-11) + 9239310
(general 12-15), scripts `run_anchor_40k_{preempt,general}.sbatch` + `_run_anchor_40k_common.sh`
(mirrors the J_40k structure). Next: old pipeline downstream (convert/QE/segale) → train →
compare BLEU/COMET/LAAL vs the top5-axis5 baseline on [[acl-6060]] + [[simul-tst-common]].

## Production follow-through (2026-07-17)

anchor_40k decode finished 40,000/40,000 (Jul 13); segale-p24 + MetricX QE completed Jul 15
(`anchor_40k-segale-p24/qe3-aligned-max`, 18,937 kept). Length-ratio filter (0.7–1.5 ref)
kept **18,932 / 18,937** — median ratio 1.036, mean 1.042 (the old consensus's long-ratio
tail is gone; the anchor is length-canonical by construction). Training chain launched:
conv2swift 9339625 (VARIANT_TAG=anchor40k, SAMPLE_N=12500) → train 9339626
(`gigaspeech-zh-consensus-anchor40k-s-bsz4`) → infer+eval launchers 9339629 ([[acl-6060]])
and 9339637 ([[simul-tst-common]], new `run_infer_after_train_simultst.sbatch`).

## Trained-model verdict (2026-07-17): NEGATIVE — degenerate over-generation sinks it

`gigaspeech-zh-consensus-anchor40k-s-bsz4/v0-20260717-141356-hf` on [[simul-tst-common]]
(seg 960/1920/2880/3840), vs the top5-axis5 baseline:

| ckpt | BLEU | COMET | LAAL (ms) |
|---|---|---|---|
| anchor40k | 14.2 / 25.3 / 29.3 / 29.9 | .783 / .825 / .852 / .845 | **35106 / 12429 / 6890 / 8815** |
| top5-axis5 | 27.5 / 32.1 / 34.1 / 34.2 | .831 / .859 / .867 / .872 | 3440 / 1534 / 2412 / 2888 |
| hibiki | 38.3 / 40.4 / 40.8 / 41.1 | .838 / .861 / .866 / .869 | 1259 / 1860 / 2498 / 2980 |

**Root cause (forensics on instances.log, correcting the initial "wait-forever" reading):**
write cadence is IDENTICAL to baseline (~310 write events/talk, median gap 1920 ms) — the
model does not wait. The real defect is **over-generation** (+19 % chars vs baseline) and
degenerate onomatopoeia loops on non-speech audio (applause/laughter/music in TED talks):
"呼！"-floods (one talk = 8,456 chars of pure 呼！), "嘘！嘘！", "谢谢。×9",
"（掌声）（欢呼声）" chains. Rep-4gram-frac: anchor 0.115 vs baseline 0.043 vs hibiki 0.010.
The floods wreck mWER resegmentation → BLEU craters AND the 12–35 s LAAL is a
**scoring-alignment artifact**, not real latency. Mechanism: training targets are clean
(0/3000 degenerate), but the anchor teaches "always fluently continue" in bigger bursts
(max commit 22.5 chars/chunk vs flagship 16.9); the consensus vote's cross-future agreement
requirement was the trained model's built-in **silence brake** on contentless audio, and the
anchor removed it (the veto doesn't exist at inference; GigaSpeech clips are speech-dense so
nothing counteracts it). ACL 6060 record incomplete: seg960 infer failed twice with vLLM
engine-core-init errors (9340889_1, 9361441_1); chained eval cancelled.

## Hybrid follow-up (2026-07-19): NEGATIVE — wording alone is worth ~nothing

`consensus_decoding_hybrid.py` (+ `run_hybrid_smoke500.sbatch`, SLURM 9361805): consensus
track decides WHEN (bit-identical vote on its own prefix → commit budget k), anchor decides
WHAT (greedy hibiki-prompt continuation, first k tokens). Same 500 utts, dual-track paired.

| | hybrid | consensus |
|---|---|---|
| char-BLEU | 60.70 | 60.55 |
| paired Δ | **+0.15** (238 W / 221 L / 41 T) | — |
| LAAL | 5.97 | 4.72 |
| 4-gram recall | 0.510 | 0.498 |

Gate was ≥ +4 → **fails decisively**. Register still partially normalizes (将 37.1 vs 63.4,
把 61.6 vs 40.7, ref 32.3/66.4) but buys no BLEU. **This falsifies the wording hypothesis:**
config A's +6.98 came from its *timing* (waiting longer → more source context before
committing, LAAL 6.55 vs 4.75), not from the anchor's wording. Combined with the trained-model
verdict (strict-veto timing is untrainable), both halves of anchor-and-veto are now
individually falsified. The hybrid's LAAL +1.25 despite identical commit budgets is
boundary-trimming deferral (anchor_trim_to_boundary re-defers trimmed tokens).

**Decision: method closed.** No 40k, no training run. The remaining ~7 BLEU gap vs hibiki on
this axis is the honest cost of ref-free decoding; further consensus quality work should
target the vote itself (e.g. context window, voter quality), not post-hoc reword/re-time.

## Related
- [[2026-07-consensus-register-forensics]] (parent fix plan), [[consensus-decoding]],
  [[future-sampling]], [[latency-quality-tradeoff]], [[synthesis-pipeline]], [[scoreboard]],
  [[comet-vs-bleu-ranking]], [[gigaspeech]].
