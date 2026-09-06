---
title: Ambiguity future-set 40k run (Qwen3.8 + Gemma-4-E2B samplers, Qwen3.6 probe) — BLEU up, COMET down
type: experiment
tags: [synthesis, consensus, ref-free, future-sampling, ambiguity, results, trained-eval, simul-tst]
sources:
  - ../codes/gigaspeech/future_sampling/ambiguity_sampler_prompt.py
  - ../codes/gigaspeech/future_sampling/consensus_decoding_token_id_level_instruct.py
  - ../codes/gigaspeech/future_sampling/run_ambiguity_q38_gemma_q36_worker.sh
  - ../codes/gigaspeech/future_sampling/external_runner/dedupe_decode_root.py
  - scripts/submit_ambiguity_40k.sh
  - scripts/resubmit_ambiguity_40k_post.sh
  - /home/haolingp/slurm_runs/ambiguity-q38-gemma-q36-fsetv2-prefixnorm-strict-40k-r1-20260831/run_manifest.txt
  - ckpts/infinisst-omni/gigaspeech-zh-consensus-ambiguity-q38-gemma-q36-fsetv2-prefixnorm-strict-40k-r1-20260831-s-bsz4/v0-20260906-013823-hf/evaluation/
created: 2026-09-05
updated: 2026-09-05
---

# Ambiguity future-set 40k run — BLEU up, COMET down

**Question.** Replace the hand-written 5-axis future sampler of [[consensus-decoding]] with a
generic [[ambiguity-future-set]] prompt (10 plausible + 10 contrastive continuations per
sampler, planned jointly), swap in newer samplers and probe, keep the strict 100 % gate.
Does trained quality move, and in which metric?

**Answer.** BLEU rises by 6–8 on [[acl-6060]] and by ~11 on [[simul-tst-common]] at every
latency ≥ 1920 ms, passing the ref-based hibiki system on both sets. XCOMET falls by
0.01–0.04. Under [[comet-vs-bleu-ranking]] this is **not yet a win** over the flagship
`top5-axis5`; it is a large surface-form gain with a small adequacy loss, and two confounds
(training-set size, probe/sampler swap) are unresolved.

## Frozen configuration (run tag `ambiguity-q38-gemma-q36-fsetv2-prefixnorm-strict-40k-r1-20260831`)

| item | value |
|---|---|
| sampler prompt | `future_set_v2_two_groups` — 10 plausible + 10 contrastive per model, one numbered list |
| samplers | `gemma-4-E2B-it` + `Qwen3.8-27B-FP8`, colocated on GPU 0 |
| probe / translator | `Qwen3.6-35B-A3B-FP8` on GPU 1 (was Qwen3-30B-A3B-Instruct-2507) |
| candidate filter | clean → validity/meta → `too_short` (<3 words) → `repeats_observed_prefix` → per-model/mode dedupe (opening word ≤1, Jaccard ≥0.65); prefix normalization case-insensitive at word boundary |
| consensus | `min_voters_ratio=1.0`, top-k 6, min-p 0, future window 1 sentence unit, min horizon 2 |
| source | old-ASR frozen TSV, rows 0–39,999 |
| post-processing | SEGALE 24 shards → MetricX QE-MAX ≤ 3.0 → length ratio 0.7–1.5 → convert (all survivors) → LoRA on [[qwen3-omni]] via [[megatron-swift]] |
| decode git commit | `13e1135` (chain rebuilt at `0874330`) |

## Pipeline counts

| stage | result |
|---|---|
| decode | 40,000 / 40,000 unique utterances (verifier: 0 missing, 0 unreadable); 1,793 duplicate copies from released main tasks 0/1 excluded via a symlink view (`-dedup`) |
| SEGALE align | 146,907 sentence rows, 24/24 shards |
| MetricX QE-MAX ≤ 3.0 | 17,326 kept (43.3 %; flagship pool was 18,599 = 46 %) |
| length ratio 0.7–1.5 | 17,306 kept (11 short, 9 long; median ratio 1.03) |
| training instances | **17,306 — all survivors, no 12,500 downsample** (submitter target 40,000 exceeded the pool) |
| training | 1,029 iters, gbs 4, 64 min on 4×L40S, final lm loss ≈ 0.70 |
| HF export | `v0-20260906-013823-hf` |

## Results — ACL 6060 dev, en→zh (seg 960 / 1920 / 2880 / 3840)

| system | BLEU | XCOMET | LongYAAL CU ms |
|---|---|---|---|
| **this run** | 37.0 / 45.7 / 47.2 / **47.8** | .748 / .787 / .796 / .798 | 1345 / 1970 / 2566 / 3031 |
| `top5-axis5` flagship | 34.9 / 39.6 / 40.1 / 40.1 | .787 / .808 / .812 / **.817** | 1461 / 2176 / 2745 / 3107 |
| hibiki (ref-based) | – / – / – / 46.8 | .780 / .812 / .814 / .820 | – / – / – / 3326 |
| EAST-even | – / – / – / 46.8 | – / – / – / .789 | – / – / – / 3533 |

chrF this run: 36.1 / 39.8 / 40.5 / 40.7.

## Results — Simul-tst-COMMON, monotonic refs (seg 960 / 1920 / 2880 / 3840)

| system | BLEU | XCOMET | LongYAAL CU ms |
|---|---|---|---|
| **this run** | 20.8† / 42.8 / 45.0 / **45.9** | .769† / .840 / .857 / .860 | 15495† / 2078 / 2442 / 2884 |
| `top5-axis5` flagship | 27.5 / 32.1 / 34.1 / 34.2 | .831 / .859 / .867 / **.872** | 3535 / 1543 / 2409 / 2855 |
| hibiki (ref-based) | 38.3 / 40.4 / 40.8 / 41.1 | .838 / .861 / .866 / .869 | 1282 / 1849 / 2429 / 2870 |
| EAST-even | 40.1 / 43.7 / 44.2 / 43.6 | .765 / .817 / .837 / .846 | 1107 / 1900 / 2620 / 3243 |

chrF this run: 28.3 / 36.7 / 38.7 / 39.0. † seg960 is degenerate (see hygiene) and should be
excluded from any comparison.

## Chunk-level SimulEval BLEU (second metric family, see [[chunk-bleu-streamlaal-scoreboard]])

| set | seg 960 / 1920 / 2880 / 3840 |
|---|---|
| ACL 6060 dev | **46.45 / 54.76 / 56.56 / 57.23** (Hibiki 46.08 / 49.17 / 51.17 / 51.85; consensus topk5 36.13 / 40.65 / – / –) |
| Simul-tst-COMMON | 27.99† / 54.71 / 57.04 / 58.14 |

## Output hygiene (instances.log; length ratio to ref, repeated-4gram fraction)

| set / seg | this run | flagship |
|---|---|---|
| ACL 960 / 1920 / 2880 / 3840 | 1.17 / 1.02 / 1.02 / 0.99 · rep .21 / .20 / .21 / .20 | 1.10 / 1.04 / 1.05 / 1.05 · rep .17 / .18 / .18 / .18 |
| tst 960 / 1920 / 2880 / 3840 | **1.89 / 1.09 / 1.01 / 1.00 · rep .30** / .16 / .14 / .12 | 1.24 / 1.10 / 1.06 / 1.06 · rep .16 / .13 / .09 / .10 |

No empty outputs anywhere. Only Simul-tst seg960 is degenerate: length ratio 1.89, rep-4gram
0.30, LongYAAL 15.5 s — the TED non-speech over-generation seen in
[[2026-07-anchor-smoke500-sweep]], here confined to the lowest latency. Everything else is
within baseline hygiene, so the BLEU gain is real surface-form change, not inflation.

## Findings

1. **The BLEU deficit vs hibiki is gone.** +7.7 (ACL) / +11.7 (tst) over the flagship at 3840,
   +1.0 / +4.8 over hibiki, at equal or lower latency. This is what
   [[2026-07-consensus-register-forensics]] predicted would happen if consensus wording moved
   onto the canonical greedy manifold — achieved here by changing the probe (Qwen3.6) and the
   candidate distribution (ambiguity futures), not by post-editing or re-timing.
2. **COMET dips 0.01–0.04**, most at ACL seg960 (.748 vs .787). Under the rank-by-COMET rule
   the flagship still wins. Whether the loss is adequacy or a training-size artifact is open.
3. **Confounds.** (a) 17,306 vs 12,500 training instances; (b) probe *and* samplers changed
   together; (c) QE survivor rate 43 % vs 46 %.
4. **seg960 timing pathology on TED** reappears (cf. anchor40k, bestof5), suggesting the new
   targets teach slightly larger bursts (max commit per chunk should be measured).

**Contradiction flagged (not overwritten).** [[2026-06-consensus-post-edit-bleu]] and
[[consensus-decoding]] state the ~7 BLEU gap is "structural, the honest cost of being
future-blind". This run closes the gap while remaining ref-free and future-blind, so the
structural claim is falsified as stated; what survives is the narrower claim that the gap
cannot be recovered by *post-hoc* editing, selection, or re-timing.

## Case-level profiles (100-case viewer bundle, first 100 rows, one recording)

- Verified 21 steps where waiting on the futures was decisive
  (`data_synthesis/reports/future_consensus_success_cases_2026-09-04.md`): READ rate ~56 %
  regardless of prefix ending; the true continuation appears among futures in only 23 % of
  steps — the method needs futures to *span* alternatives, not to be right.
- `AUD0000000003_1059` (success): "sat deep" — futures split physical/mental, probe split
  坐在 vs 沉思/陷入, empty intersection → READ; correct 国王陷入沉思 after "in thought".
- `AUD0000000003_1011` (failure): "They were so huge that the" — 36/36 futures' distributions
  put 它们 first (pF = 0), probe bias, not lack of diversity.
- `AUD0000000003_1125` (failure): at chunk 4 the vote correctly split 他/她 (9 Qwen futures
  carried "her/she", 0 Gemma) → READ; after the sentence-unit prefix reset the futures no longer
  mention the person, 36/36 default to 他 → WRITE. Antecedent row `_1124` is absent from the
  frozen TSV. Mechanisms: data boundary, reset-changes-the-question, probe gender prior.

## Next

- Matched 12,500-instance rerun (seed 42) to remove confound (a).
- Per-sentence COMET diff vs flagship on tst seg3840 (July forensics recipe).
- Inspect tst seg960 outputs; consider an inference-time repetition brake.
- Sampler ablations on an ambiguity-stratified set (Qwen-only 20/40, + larger Gemma-4 12B/31B):
  metric = fraction of ambiguity steps where a sampler produces ≥1 cue-carrying future.
- Reset-aware future window: sample from uncommitted source + current unit, capped at 2 units.

## Related
- [[ambiguity-future-set]], [[consensus-decoding]], [[future-sampling]], [[scoreboard]],
  [[comet-vs-bleu-ranking]], [[simul-tst-common]], [[acl-6060]], [[latency-quality-tradeoff]],
  [[2026-06-consensus-axis5-vs-futures200]], [[2026-07-consensus-register-forensics]],
  [[2026-07-present-propose-gate]], [[2026-07-anchor-smoke500-sweep]], [[synthesis-pipeline]],
  [[qwen3-omni]], [[babel-cluster]].
