---
title: Present-ranked consensus winner (register fix #2, ref-free)
type: experiment
tags: [consensus, register, ref-free, bleu, winner-rule, synthesis]
sources:
  - ../codes/gigaspeech/future_sampling/experimental/consensus_decoding_present_rank.py
  - ../codes/gigaspeech/future_sampling/experimental/run_present_rank_smoke500.sbatch
created: 2026-07-19
updated: 2026-07-19
status: NEGATIVE — smoke500 killed at -2.15 paired char-BLEU (2026-07-19)
---

# Present-ranked consensus winner (register fix #2, ref-free)

Successor to [[2026-07-refsel-bestof5]] (best-of-N ref selection: +10.9 oracle → ~0
trained BLEU, so **per-utt selection does not transfer**). This attacks the gap with the
only lever training demonstrably absorbs: a **systematic token-level distribution shift**
— the trained flagship reproduces the register bias of its targets exactly (never says
但是, over-uses 因此; [[2026-07-consensus-register-forensics]]).

## Hypothesis

The forensics identified the future-averaged winner rule as the register amplifier:
formal connectives survive under more sampled continuations, so argmax-over-futures
systematically picks them, dragging targets ~40 BLEU off the canonical greedy manifold
that both the frozen reference and hibiki's targets live on. Fix plan **#2
(rank-by-present)** was never run — anchor-and-veto (fix #1) failed for *timing* reasons
(train-time over-generation), not wording reasons.

## Design (fully ref-free)

Decoder `experimental/consensus_decoding_present_rank.py` (monkeypatch of the flagship
module, CLI-identical):
- **Gate unchanged** — strict intersection of every future's top-6 candidates, exactly
  the J_40k prod config (verbose header `num_futures=20, top_k=6, min_p=0.0`; old
  strict `choose_consensus_token`, pre-c5647b7). READ/WRITE policy therefore stays
  flagship-familiar — the anchor failure mode (timing shift) is designed out.
- **Winner changed** — among gate-eligible tokens, argmax of the **present
  distribution**: the same probe prompt conditioned on observed source only, NO future
  appended. Rides as prompt 0 in the same batched `/completions` call (+1 prompt/step).
  Futures-mean is only the tie-break for tokens the present probe didn't score
  (`present_fallback` logged).
- Step meta logs `baseline_token` vs winner (`changed_vs_baseline`) for forensics.

## Smoke500 protocol (job 9368591)

Rows 0–499 of the flagship frozen-ref TSV, single track; **paired against the J_40k
per-utt jsons on disk** for the same utt_ids (`analyze_present_rank_smoke.py`, needs the
evaluation env for jieba). Leading indicators (forensics fix plan #4):
- paired char-BLEU vs frozen ref: 59 → **65+ hoped, ≥+2 to proceed**
- 4-gram recall vs ref: 0.48 → 0.6+
- marker profile (因此/但是/所以…) moves toward the ref column
- **Hygiene gates (hard, per Haoling 2026-07-19):** length_ratio_ref → 1.0 with no
  fatter >1.5 tail than J_40k; rep-4gram not above J_40k (degeneration guard);
  LAAL p50/p90/p99 within family of J_40k (timing guard).

Kill: paired delta < +2, or any hygiene gate fails → file negative result; the
winner-rule family is then exhausted and remaining ref-free levers are prompt-register
guidance in the probe (fix #4 cousin) or accepting [[comet-vs-bleu-ranking]] as the
paper story.

## Result (2026-07-19): NEGATIVE — killed by the smoke gate

Full 500/500 paired vs J_40k (job 9368591, 2h47m wall):

| metric | present-rank | J_40k | delta |
|---|---|---|---|
| char-BLEU (mean) | 57.51 | 59.66 | **−2.15** |
| paired delta median / W-L-T | | | −1.77, 196/277/27 |
| LAAL p50/p90/p99 | 5.16/7.91/10.46 | 5.12/7.76/10.62 | matched |
| 4-gram recall vs ref | 0.464 | 0.484 | −0.02 |
| rep-4gram / lr tails | 0.004, 1×lr>1.5 | 0.003, 0 | clean |
| 因此 / 将 / 把 per 10k | 9.7 / 61.3 / 41.2 | 9.9 / 63.8 / 35.3 | ref: 5.8 / 32.3 / 66.4 |

Three-point reading:
1. **The winner rule has no freedom.** Register markers are statistically unmoved
   (因此 9.7 vs 9.9; 将 61.3 vs 63.8 — both still 2× the ref's 32.3). The strict
   top-6 intersection typically leaves 1–2 eligible tokens, so ranking them by the
   present distribution can't change the register. **Register bias enters at the
   GATE (the future-conditioned candidate sets), not at the winner rule.**
2. Where the rules did diverge, present won slightly less often (196/277) —
   present-myopia is real (n=1 forensic: garbled ending 53.3 where J_40k hit the ref
   verbatim at 89.2). The future-averaged winner is the better rule *given this gate*.
3. Hygiene/timing held perfectly (the design goal of keeping the gate) — so the
   negative is a clean measurement of the winner rule alone.

Prompt forensics (same day): the probe prompt and the frozen-ref prompt share
near-identical [TASK]/[INPUT] blocks (diff = probe's [IMPORTANT] continuation
mechanics vs ref's "Output only the Chinese translation."). So prompt misalignment
is NOT the register source either. Remaining suspect: **the future text appended
into [INPUT]** shifts every future-conditioned distribution toward written register,
and the gate is the intersection of exactly those distributions.

→ Successor: [[2026-07-present-propose-gate]] — invert the roles: present
distribution PROPOSES the candidate set, futures only VERIFY (majority support).

If smoke passes: 40k decode (clone `_run_J_40k_common.sh` with the experimental
decoder) → SEGALE + QE → convert (SAMPLE_N=12500 seed 42) → train → eval vs top5-axis5
+ hibiki on [[acl-6060]] + Simul-tst-COMMON. Success bar unchanged: trained BLEU ≥ +2
over top5-axis5, COMET and LAAL held.

## Context: why not other ideas

- Best-of-N / MBR / QE-rerank = per-utt selection → proven no-transfer
  ([[2026-07-refsel-bestof5]]).
- Post-edit (loose & strict) → [[2026-06-consensus-post-edit-bleu]] triple negative.
- Anchor-and-veto → trained-model degeneration ([[2026-07-anchor-smoke500-sweep]]).
- Qwen3.6-35B sampler → no quality gain over Qwen3.5-122B on the 10-case debug set
  (56.9 vs 59.5 think / 57.7 no-think char-BLEU) and ~30 min/case with thinking — not
  scalable to 40k; dropped 2026-07-19.

## Related
[[2026-07-consensus-register-forensics]], [[2026-07-refsel-bestof5]],
[[consensus-decoding]], [[scoreboard]], [[synthesis-bleu-not-predictive]],
[[comet-vs-bleu-ranking]].
