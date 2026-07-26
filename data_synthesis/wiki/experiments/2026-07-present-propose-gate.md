---
title: Present-proposes / futures-verify gate (register fix #3, ref-free)
type: experiment
tags: [consensus, register, ref-free, bleu, gate, synthesis]
sources:
  - ../codes/gigaspeech/future_sampling/experimental/consensus_decoding_present_propose.py
  - ../codes/gigaspeech/future_sampling/experimental/run_present_propose_smoke500.sbatch
created: 2026-07-19
updated: 2026-07-19
status: smoke500 launched (job 9370692)
---

# Present-proposes / futures-verify gate (register fix #3, ref-free)

Successor to [[2026-07-present-rank-winner]], whose negative result relocated the
register bias: with the flagship strict gate kept, re-ranking survivors by the
present distribution moved nothing (−2.15 paired char-BLEU, markers unmoved),
because the intersection of 20 future-conditioned top-6 sets typically leaves 1–2
tokens. **The bias enters at the gate** — every candidate set is conditioned on an
appended sampled future (probe forensics: the prompts themselves are near-identical,
so the future text in [INPUT] is the shift), and the flagship's candidate universe
is the intersection of exactly those shifted sets.

## Design (fully ref-free)

`experimental/consensus_decoding_present_propose.py` — invert who authors the wording:

- **Present PROPOSES**: candidate set = top-6 of the present distribution (probe on
  observed source only, no future) — the canonical-register manifold both the frozen
  ref and hibiki's targets live on.
- **Futures VERIFY**: a proposal survives iff it appears in the top-6 of ≥
  `ceil(0.75 × 20) = 15` future-conditioned distributions (`--min-voters-ratio 0.75`,
  plumbed through the base CLI). Futures keep their real job — vetoing present-myopia
  and premature commits (the n=1 garbled-ending forensic) — without authoring tokens.
- **Winner** = argmax present probability among survivors.
- No survivor → no_consensus → READ; stopping semantics unchanged.

Not a rerun of the failed soft-vote loosening ([[2026-06-consensus-axis5-vs-futures200]],
−2..4): that relaxed the threshold over the SAME futures-proposed universe; here the
proposal universe itself moves to the present manifold.

Risk watch: majority-verify is looser than full intersection → acceptance rate may
rise → longer pending runs, LAAL/length drift (what killed anchor). The hygiene gates
below are the tripwire; if LAAL drifts, raise min-voters-ratio (0.9 → 18/20) before
concluding.

## Smoke500 protocol (job 9370692)

Identical harness to fix #2: rows 0–499 of the flagship frozen-ref TSV, paired vs
J_40k per-utt jsons (`analyze_present_rank_smoke.py`, evaluation env). Gates:
- **Proceed:** paired char-BLEU delta ≥ +2 AND hygiene clean.
- **Hygiene (hard, per Haoling 2026-07-19):** length_ratio_ref → 1.0, no fatter >1.5
  tail than J_40k; rep-4gram ≤ J_40k; LAAL p50/p90/p99 in family of J_40k.
- **Leading indicator that the mechanism works:** marker profile moves toward the ref
  column (因此 9.9→~6, 将 63.8→~40, 把 35.3→~55 per 10k) and 4-gram recall 0.484→0.55+.
- **Kill:** delta < +2 or hygiene fail. If killed with markers UNMOVED, the register
  is upstream of the candidate sets entirely (in the shared probe conditioning) and
  the decode-time family is exhausted → accept [[comet-vs-bleu-ranking]] as the story.

If smoke passes: 40k decode → SEGALE + QE → convert (SAMPLE_N=12500 seed 42) → train
→ eval vs top5-axis5 + hibiki. Success bar: trained BLEU ≥ +2 over top5-axis5, COMET
and LAAL held.

## Failed-alternatives ledger (do not rerun)

- Per-utt selection (bestof5 AND bestof4): conclusively no transfer, timing blow-ups
  ([[2026-07-refsel-bestof5]]).
- Present-rank winner over strict gate: −2.15, no freedom ([[2026-07-present-rank-winner]]).
- Anchor-and-veto: trained degeneration ([[2026-07-anchor-smoke500-sweep]]).
- Post-edit: triple negative ([[2026-06-consensus-post-edit-bleu]]).
- Soft-vote loosening (futures-proposed majority): −2..4.
- Qwen3.6-35B thinking sampler: no gain, ~30 min/case; dropped.

## Related
[[2026-07-present-rank-winner]], [[2026-07-consensus-register-forensics]],
[[consensus-decoding]], [[scoreboard]], [[comet-vs-bleu-ranking]].
