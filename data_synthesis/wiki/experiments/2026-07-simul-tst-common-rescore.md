---
title: Re-scoring zh checkpoints on Simul-tst-COMMON (monotonic refs)
type: experiment
tags: [eval, monotonic, consensus, hibiki, bleu, comet]
sources:
  - scripts/infer/infer_slurm_simultst.sh
  - scripts/infer/eval_all_ckpts_simultst.sh
  - ../simul_tst_common/
created: 2026-07-11
updated: 2026-07-11
---

# Re-scoring zh checkpoints on Simul-tst-COMMON (monotonic refs)

**Question.** Is consensus's 6–8 BLEU deficit vs hibiki on [[acl-6060]] (COMET tied) a
reference-style artifact — offline references rewarding reordering that ref-based methods
mimic — or a real quality gap? Test: re-score on [[simul-tst-common]] (interpreter-style
**monotonic** references, our 2026-07-11 rebuild, diagnostic-grade).

**Setup.** Jobs 9214519–22 (inference, seg960–3840, Standard prompt, 2×L40S) + 9214529
(omnisteval longform + XCOMET-XL vs tst.{yaml,en,zh}). 27 TED talks, 2,853 sentences.
`consensus-topk5_f200` could not run — its checkpoint was **deleted from disk**
(see [[scoreboard]] deleted-ckpt gotcha).

## Results (BLEU / XCOMET / LongYAAL-CU ms, seg960 | 1920 | 2880 | 3840)

| ckpt | BLEU | COMET | YAAL (CU) |
|---|---|---|---|
| hibiki word-align | **38.3 / 40.4 / 40.8 / 41.1** | 0.838 / 0.861 / 0.866 / 0.869 | 1282 / 1849 / 2429 / 2870 |
| consensus top5-axis5 | 27.5 / 32.1 / 34.1 / 34.2 | 0.831 / 0.859 / **0.867 / 0.872** | 3535 / 1543 / 2409 / 2855 |
| EAST-even | 40.1 / **43.7 / 44.2 / 43.6** | 0.765 / 0.817 / 0.837 / 0.846 | 1107 / 1900 / 2620 / 3243 |
| PA-40k | 25.1 / 32.3 / 34.0 / 33.6 | 0.805 / 0.847 / 0.859 / 0.855 | 7274 / 2570 / 2491 / 2756 |

(Absolute XCOMET values are higher than on [[acl-6060]] — different domain; compare only
within this set.)

## Findings

1. **The BLEU gap survives monotonic references.** At matched latency (seg1920+, YAAL
   ≈1.5–2.9 s) hibiki leads consensus by **6.5–8.3 BLEU** — the same 6–8 gap as on ACL 6060.
   The reference-style-bias hypothesis is **not supported**: the deficit behaves like a real
   surface-form/quality gap, not an artifact of offline reordering-friendly refs.
2. **COMET is tied again** (consensus even edges ahead at seg2880/3840: 0.867/0.872 vs
   0.866/0.869) — exactly the ACL 6060 pattern reproduced on an independent test set.
3. **The EAST-even canary did NOT crater.** Predicted to collapse if its ACL BLEU lead came
   from reordering; instead it still leads BLEU (43.6–44.2) while posting by far the worst
   COMET (0.765 at seg960). So either EAST's BLEU advantage never came from reordering, or
   longform BLEU is simply insensitive to order at this granularity — both readings reinforce
   [[comet-vs-bleu-ranking]] (BLEU anti-correlates with adequacy at the top of this table).
4. Consensus's seg960 point is degenerate (YAAL 3535 ms — higher than its own seg2880/3840);
   its low-latency operating point remains the weak spot.

**Caveat.** Refs are diagnostic-grade: only 37 % of NAIST's interpreter QC edits applied
(GPT-4o drift), so 63 % of lines are raw Salami GPT-4o output. They are still
monotonic-by-construction (chunk-wise translation), but stylistically GPT-flavored — a
paper-grade rerun needs the authors' final file (email drafted).

**Consequence.** Treat the consensus-vs-hibiki BLEU deficit as **real** until the
official refs say otherwise → synthesis levers back on the table: paraphrased-oracle
futures diagnostic, Qwen-122B future sampler ([[qwen35-122b-sampler]]).

## Forensics: where the ~7 BLEU lives (seg3840, sentence-level)

Per-sentence diff of the two models' resegmented outputs (2,853 aligned pairs; scripts in
session scratchpad, records reproducible from `instances.resegmented.jsonl`):

- **Tail-heavy:** median delta only +2.8 chrF, but the worst 10 % of sentences carry
  **71 %** of the total gap mass. Hibiki wins 53 % of sentences, consensus 25 %, tie 22 %.
- **Not GPT-flavor matching (confound refuted):** the gap is identical on
  interpreter-edited reference lines (+5.1 chrF, n=703) and raw-GPT lines (+4.9, n=2150).
  Hibiki's edge holds against human-corrected text just as strongly.
- **Canonicality, not adequacy:** hibiki frequently emits the *canonical* rendering
  verbatim (sent-chrF=100 on many short refs: 我是汤姆 / 有很多人想去那里), while consensus
  emits correct but *marked* variants (我叫汤姆 / 大量希望前往那里的人). Register markers:
  consensus overuses literary forms (此 3×, 正是 7× the reference rate); length ratio
  1.08× vs hibiki's 1.02×. COMET ties because both are adequate.
- **Pervasive but worst on short sentences:** delta +6.3 chrF at ref-len 9–15 chars
  (hibiki verbatim-match 4–11 % there) yet still +4.1 at 26–45 chars.
- **Streaming artifacts are shared, not differential:** dangling-fragment rate ~11–12 %
  for BOTH systems (reference 0.6 %) — a longform-resegmentation cost, not consensus's.

**Key inference.** Short sentences (≤ ~3 s audio) are fully revealed before commit at
seg3840, so future-blindness is *not binding* there — yet consensus is still non-canonical.
Combined with hibiki's targets being word-aligned **frozen offline LLM translations**
(`…frozen_llm_reference.tsv`, see [[gigaspeech]]), the deficit localizes to the **style of
consensus's synthesis targets** (base-model/instruct candidate distribution + agreement
machinery), not to the honest online constraint alone. This *partially contradicts* the
"structural, future-blind cost" framing in [[consensus-decoding]] /
[[2026-06-consensus-post-edit-bleu]] — post-edit of committed text still can't fix it
(proven), but a **canonical-register candidate prior inside the commit loop** (e.g. the
frozen-reference translator as the instruct scorer, or plain-register constrained ranking)
is now the most promising untested lever, alongside the oracle-future diagnostic which
separates future-quality headroom from mechanism headroom.

## Related
- [[simul-tst-common]], [[acl-6060]], [[comet-vs-bleu-ranking]], [[scoreboard]],
  [[consensus-decoding]], [[checkpoint-evaluation]], [[latency-quality-tradeoff]].

## Sources
- scores: `<ckpt>/evaluation/simul_tst_common/en-zh/seg<N>/segmentation_output/scores.tsv`
- scripts: `scripts/infer/{infer_slurm_simultst.sh, eval_all_ckpts_simultst.sh, ckpts_simultst.txt}`
