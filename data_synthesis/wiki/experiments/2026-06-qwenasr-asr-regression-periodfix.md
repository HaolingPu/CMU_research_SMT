---
title: New qwenasr ASR regresses vs the old-asr 5-axis baseline; period-fix is not the cure
type: experiment
tags: [synthesis, consensus, asr, regression, period-fix, ablation, results, trained-eval]
sources:
  - ../codes/gigaspeech/future_sampling/period_fix_traj_nested.py
  - ../codes/gigaspeech/future_sampling/scripts/segale/submit_J40k_post.sh
  - scripts/train/convert2swift_consensus.py
  - ckpts/infinisst-omni/gigaspeech-zh-consensus-FULL40k-win3-s-bsz4/
  - ckpts/infinisst-omni/gigaspeech-zh-consensus-FULL40k-win3-nopfix-s-bsz4/
  - ckpts/infinisst-omni/gigaspeech-zh-consensus-FULL40k-win3-splitfix-s-bsz4/
  - ckpts/infinisst-omni/gigaspeech-zh-consensus-top5-axis5-s-bsz4/
  - ckpts/infinisst-omni/gigaspeech-zh-consensus-FULL40k-win3-clausesplit-s-bsz4/
  - ../codes/gigaspeech/split_src_text_full_spacy.py
  - ../codes/gigaspeech/split_src_text_full_punct.py
created: 2026-06-26
updated: 2026-06-30
---

# New qwenasr ASR regresses vs the old-asr 5-axis baseline; period-fix is not the cure

**Question.** The production baseline is the old-asr **5-axis** consensus model
([[2026-06-consensus-axis5-vs-futures200]], the COMET leader on [[scoreboard]]). We re-ran the
source audio through a **new ASR** (Qwen-ASR + spaCy sub-sentence split, "qwenasr sentsplit")
hoping for a quality bump. Instead trained eval **regressed**. Two diagnoses were tested: was it
the new data's **句号 (sentence-period) artifact** — fixable by period-fix — or the ASR data
quality itself? This page settles it with a controlled period-fix ablation.

**Answer.** The new ASR is genuinely worse data: **−4 to −6 BLEU / −0.05 COMET at every latency**,
and **period-fix recovers only +1–2 BLEU** — nowhere near closing the gap. The regression root
cause is **ASR/segmentation quality, not the period artifact**. QE filtering and period-fix both
help on the margin but cannot make the new ASR match the old.

## ASR lineage (what "old" vs "new" mean here)
- **OLD ASR** = `train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv` (original non-Qwen ASR).
  Decoded as the prod root `consensus_decoding_prod/J_40k` (window=1), QE-filtered via
  [[segale-alignment]]+[[metricx]] to `J_40k-segale-p24/qe3-aligned-max-full` (18,599 survivors) →
  trained as **`consensus-top5-axis5`**. **This 5-axis checkpoint IS the canonical old-asr+QE
  baseline** the user compares everything against. (Gotcha: the *current* default INPUT_TSV in
  `_run_J_40k_common.sh` is `qwenasr_filtered`, but that default was changed *after* J_40k was
  decoded — J_40k's actual audio source is the old asr. Do not infer J_40k's ASR from the live
  script default.)
- **NEW ASR** = `train_xl_case_robust_qwenasr_sentsplit_frozen_llm_reference.tsv` (Qwen-ASR + spaCy
  sub-sentence split). Decoded as `J_40k_qwenasr_sentsplit_win3_FULL40k` (window=3, win3 dualbase),
  the full canonical pipeline (decode 40k → period-fix → SEGALE QE-MAX≤3 → length filter → sample
  12.5k → train), see [[consensus-decoding]] and [[dataset-conversion-pipeline]].
- The new sentsplit ASR introduces a **chunk-start period artifact**: a `target_trajectory` delta
  begins with `。！？` (3.47% of deltas), an artifact of the sub-sentence boundaries.

## The three configs (all QE-filtered)

| config | checkpoint | decode | ASR |
|---|---|---|---|
| ① old asr + QE | `consensus-top5-axis5` | 5-axis, win1 | old (asr_filtered) |
| ② new asr + QE + **pfix** | `consensus-FULL40k-win3` | win3 | new (qwenasr sentsplit) |
| ③ new asr + QE + **no-pfix** | `consensus-FULL40k-win3-nopfix` | win3 | new (qwenasr sentsplit) |

## Results (trained, ACL6060 dev en→zh — BLEU / COMET)

| config | seg960 | seg1920 | seg2880 | seg3840 |
|---|---|---|---|---|
| **① old asr + QE** (5-axis) | **34.9 / .787** | **39.6 / .808** | **40.1 / .812** | **40.1 / .817** |
| ② new asr + QE + pfix (old split) | 30.8 / .741 | 35.5 / .761 | 34.8 / .752 | 34.2 / .765 |
| ③ new asr + QE + no-pfix (old split) | 29.9 / .731 | 33.0 / .754 | 33.9 / .755 | 33.8 / .763 |
| ④ new asr + QE + pfix (**spaCy split-fix**) | 30.8 / .744 | 35.4 / .768 | 35.5 / .771 | 36.7 / .764 |
| ⑤b new asr + QE + pfix (**clause-split**) | 29.4 / .743 | 34.3 / .770 | 34.2 / .765 | 33.1 / .762 |

**Regression (② − ①), even with period-fix on:** ΔBLEU −4.1 / −4.1 / −5.3 / −5.9;
ΔCOMET −.046 / −.047 / −.060 / −.052.

## The split-fix attempt (④): clean A/B vs ②, only the sub-sentence split differs

The original qwenasr sub-sentence split (`split_src_text_full_spacy.py`) cut BEFORE every
coordinating conjunction on a comma, leaving 1-content-word garbage fragments (`Editor,`, `and.`,
`so.`). Hypothesis: this over-segmentation, not transcription, drives the residual gap. The fix
(`_merge_short(min_words=2)`) folds sub-2-content-word fragments into a neighbour, then re-decodes
the **identical** 40k rows through the **identical** win3 + pfix + QE-MAX≤3 + length + sample-12.5k
+ train pipeline. So ④ vs ② isolates the split alone (QE survivors 16,783; checkpoint
`consensus-FULL40k-win3-splitfix`).

**Split-fix contribution (④ − ②):** ΔBLEU +0.0 / −0.1 / **+0.7** / **+2.5**;
ΔCOMET +.003 / +.007 / **+.019** / −.001. **Real but latency-skewed** — the gain lands at the long
segments (seg2880/3840), exactly where mid-clause fragments hurt streaming alignment most; the short
segments are unchanged. It narrows the seg3840 regression from −5.9 to **−3.4** BLEU.

**Still does NOT close the gap (④ − ①):** −4.1 / −4.2 / −4.6 / −3.4 BLEU. The split fix recovers
a slice of the long-latency loss but ~3–4.6 BLEU of regression remains.

## The clause-split attempt (⑤b): granularity-matching hypothesis — REFUTED

Finding #6 proposed re-segmenting on sentence-final punctuation to kill the mid-clause cuts entirely.
A pure sentence-split (⑤) was clean but coarser than old asr (3.93 vs 5.48 units/utt), so we built a
**clause-level** splitter (`split_src_text_full_punct.py --mode clause`, no spaCy): split on
sentence-final punct `[.!?;。！？；]` with abbreviation/decimal guards, then at comma+clause-conjunction
junctions, keeping the comma with the left clause — old-asr style. This matched old asr almost
exactly: **5.3 vs 5.1 units/utt, 11.9 vs 12.3 words/unit, 1-word frags 0.01% vs 1.05%.** Re-decoded
the identical 40k rows through the identical win3 + pfix + QE-MAX≤3 + length + sample-12.5k + train
pipeline (checkpoint `consensus-FULL40k-win3-clausesplit`, HF `v0-20260630-060135`).

**Result — clause-split does NOT help; it slightly regresses BLEU vs the spaCy split-fix:**

| Δ | seg960 | seg1920 | seg2880 | seg3840 |
|---|---|---|---|---|
| ⑤b − ④ (vs spaCy split-fix) | −1.4 / −.001 | −1.1 / +.002 | −1.3 / −.006 | **−3.6** / −.002 |
| ⑤b − ② (vs old split) | −1.4 / +.002 | −1.2 / +.009 | −0.6 / +.013 | −1.1 / −.003 |
| ⑤b − ① (vs old-asr baseline) | **−5.5** | **−5.3** | **−5.9** | **−7.0** |

**The granularity-matching hypothesis is refuted.** ⑤b matched old-asr clause granularity almost
perfectly, yet BLEU is *worse* than both ② and the "buggy" spaCy ④ (−1 to −3.6 BLEU; COMET flat),
and the gap to ① is **unchanged at −5.5..−7.0 BLEU** — actually the *widest* seg3840 gap of any new-asr
config. Cleaning the split further did not recover anything; it cost BLEU. → **the residual qwenasr
regression is NOT explained by segmentation boundary/granularity at all.** The boundary-leakage audit
(below) correctly measured that new asr cuts mid-clause, but *fixing* that (④→⑤b) does not move eval,
so leakage is at most a minor contributor, not the root cause. The remaining suspects are **non-split**
factors: the decode window (**win3 vs the baseline's win1** — see open question), the qwenasr
transcription *content* itself, or the frozen-reference pairing.

**Caution on ④'s seg3840 = 36.7.** That single cell now looks like an outlier: it beat ② by +2.5 and
⑤b by +3.6 at seg3840 while every other ④ cell was ≈②. A cleaner, same-granularity split (⑤b) did
not reproduce it. Treat ④'s long-latency "win" as likely noise / a lucky checkpoint rather than a real
split effect — the clause-split A/B is the stronger test and it shows no split gain.

## Why the new ASR still trails — segmentation boundary leakage (not transcription)

A 401-utterance reproducible-sample audit of the SAME shared utterance ids (old vs new
`src_text_full`, see scratchpad `analyze.py`/`patterns.py`/`boundary.py`/`overlap.py`) shows the
residual gap is **not** mis-transcription:
- **Per-utterance transcription quality is fine.** Word-level WER-proxy vs old: mean **2.7%**,
  median 2.3%, p99 ~9.7%; 98.9% of utterances ≤10%. No truncation (0/401 >20% shorter), no length
  drift (total char ratio 1.012), no hallucination (repeated-word 8.2% new vs 10.0% old). Casing is
  *better* in new (ALL-CAPS tokens 2.2% new vs 22.2% old); numbers are digits (`1990s`) vs spelled
  (`nineteen ninety s`). None of these favour the old asr.
- **The damage is segmentation boundary leakage.** New still cuts the same audio ~**1.47×** finer
  (26.6 vs 18.0 segments/file; 78% of files get more segments) and the cuts land **mid-clause**:
  - **TAIL-leak 21.2%** — a segment ends on a dangling incomplete clause from the *next* segment
    with a period stapled on: `"...were better, but another."`, `"...the house again, and she."`,
    `"...in accordance with."`, plus many `"...but."` / `"...and."` / `"...he was."`.
  - **HEAD-leak 15.0%** — a segment starts with a leaked fragment from the *previous* segment:
    `"Came in, leading..."`, `"Kingdom, the death of Lord Marney..."`, `"Editor, and these..."`.
  - **Boundary duplication** — 9.7% of adjacent segment pairs share a 2-word overlap (the leaked
    fragment appears at the tail of N *and* the head of N+1), so the same words get translated
    twice in adjacent training pairs. Verified verbatim: `_805` ends `"...ago in accordance with"`,
    `_806` starts `"in accordance with the custom prevailing..."`.
  - Old asr's differences are mostly mid-sequence word-choice variation, not boundary garbage.

**Mechanism:** the frozen-LLM `llm_reference_text` is forced to translate a hanging `"...but."` /
`"and she."` / `"in accordance with"`, producing truncated or awkward Chinese targets, and the model
learns to emit incomplete continuations + boundary garbage. This is the clean cause of a 3–4.6 BLEU
regression that the ~2.7% WER would otherwise hide — and it explains why `min_words=2` (which only
kills the *visible* 1-word fragments) recovered the long-latency slice but not the bulk: the
**mid-clause cut + duplication** survives the merge. To close the rest, re-segment on sentence-final
punctuation only (no mid-clause cc-cuts) or merge dangling ≤3-word boundary fragments back into the
neighbour AND de-dup the boundary overlap, before regenerating references.

## The period-fix ablation (the controlled experiment)

Cleanly isolates period-fix because it only rewrites `target_trajectory` (moves a leading `。` to
the previous delta's end via `period_fix_traj_nested.py:fix_traj`); it **never touches
`prediction`**, which is what SEGALE/QE/length filtering score. So ② and ③ share the **identical
16,979 QE+length survivor set** and **identical conv2swift sampling** (same seed 42, same sorted
order → same rng → same 12,446 instances, same multipliers, same audio clips). **The only variable
is the period placement in the training text.** 39% of survivor trajectories (6,627) actually
differ → enough signal.

**Period-fix contribution (② − ③):** ΔBLEU +0.9 / +2.5 / +0.9 / +0.4 (avg **+1.2**);
ΔCOMET +.010 / +.007 / −.003 / +.002 (~flat). Real but small.

## Same-method corroboration (rules out "it's the win3 method's fault")

①(5-axis@win1) vs ②③(win3) confounds ASR source with decode method. The **5-axis method was also
run on qwenasr**, isolating ASR within one method:

| 5-axis, same method | seg960 | seg1920 | seg2880 | seg3840 |
|---|---|---|---|---|
| old asr (`top5-axis5`) | 34.9 / .787 | 39.6 / .808 | 40.1 / .812 | 40.1 / .817 |
| qwenasr + pfix (`…-qwenasr-fixed`) | 31.9 / .743 | 36.1 / .777 | 34.3 / .760 | 34.7 / .769 |
| qwenasr no-pfix (`…-qwenasr`) | 29.5 / .732 | 32.3 / .749 | 32.9 / .746 | 32.7 / .755 |

Switching to qwenasr drops **−3 to −7 BLEU** under the *same* 5-axis method — same magnitude as the
win3 drop. → the regression travels with the **ASR data**, not the decode method. Period-fix again
buys ~+2–4 BLEU but doesn't close it. (Note: the `-qwenasr` 5-axis runs use `qwenasr_filtered`, not
`sentsplit`; the win3 runs use `sentsplit`. Both qwenasr variants regress.)

## Findings

1. **The new qwenasr ASR is the regression root cause** — a structural −4–6 BLEU / −0.05 COMET data
   penalty vs old asr, reproduced across two decode methods (win3 and 5-axis).
2. **Period-fix is NOT the cure.** The 句号 artifact it removes is worth only ~+1.2 BLEU / ~0 COMET.
   It's a real, cheap polish (keep it on), but it does not explain or fix the regression. This
   **corrects the earlier hypothesis** that the period artifact was the main driver.
3. **QE-MAX filtering does the heavy lifting against catastrophic collapse**, but not against the
   old↔new gap. (On an unfiltered new-asr control, seg960 collapsed to BLEU 6.9; QE filtering alone
   restored it to ~30 — a +23 BLEU rescue — see the QE pass in [[metricx]]/[[segale-alignment]].
   But filtered new-asr still sits 4–6 BLEU below old asr.)
4. **The sub-sentence split fix (④) is real but partial.** Merging sub-2-word fragments
   (`_merge_short`) recovers +0.7/+2.5 BLEU and +.019 COMET at long latencies (seg2880/3840),
   narrowing seg3840 from −5.9 to −3.4, but leaves ~3–4.6 BLEU of regression.
5. **Boundary leakage is real in the data but is NOT the eval driver — refuted by ⑤b.** The audit
   correctly found qwenasr cuts ~1.47× finer and mid-clause (HEAD-leak 15% / TAIL-leak 21% / 9.7%
   boundary dup). But a clause-split (⑤b) that matched old-asr granularity almost exactly (5.3 vs 5.1
   units/utt) and eliminated those fragments **did not recover any eval** — it lost −1..−3.6 BLEU vs
   the messier ④ and stayed −5.5..−7.0 vs ①. So leakage/granularity is at most a minor contributor.
   This **corrects finding-6-as-written**: cleaning the split further was the wrong lever.
6. **Segmentation is not the root cause.** Across ④(spaCy merge) and ⑤b(clause), the split changed a
   lot and eval barely moved (and moved the *wrong* way on BLEU). The −5–7 BLEU regression is
   **decode-window or transcription-content driven, not split-driven.** ④'s lone seg3840=36.7 is best
   read as noise (⑤b did not reproduce it under a cleaner same-granularity split).
7. **Sharpened open question — win1 vs win3 (the last uncontrolled confound).** ① baseline decodes at
   **win1** (`--future-source-window-chunks=1`); every new-asr config (②③④⑤b) decodes at **win3**.
   Now that split is ruled out, the window is the largest un-isolated variable between ① and the new
   runs. The clean test: re-decode a fixed new-asr manifest (⑤b clausesplit) at **win1** and compare
   to its win3 self — if win1 recovers a big slice, win3 was a real confound; if not, the residual is
   pure qwenasr transcription content. See "Open: win1 vs win3" below.

## Open: win1 vs win3 (in progress)

The one variable never isolated between ① (34.9–40.1 BLEU) and every regressed new-asr run is the
**consensus decode future-source window**: ① used `win1`, ②③④⑤b all used `win3`
(`--future-source-window-chunks`, see [[consensus-decoding]]). The same-method 5-axis corroboration
(above) rules out win3-vs-5-axis *method* but NOT win1-vs-win3 *window* — both qwenasr 5-axis runs and
all win3 runs regress, yet none of them is a win1 run on qwenasr, and ① is the only win1 run and also
the only old-asr run, so ASR-source and window are still perfectly confounded at win1.

**Test being run:** re-decode the fixed ⑤b clausesplit manifest at **win1**, identical downstream
(pfix → QE≤3 → length → sample 12.5k → train → 4-seg eval), so `win1-clausesplit − ⑤b(win3)` isolates
the window alone on new asr. Outcomes: (a) win1 ≫ win3 → the "regression" was substantially a decode
setting, revisit whether new asr is actually worse; (b) win1 ≈ win3 → window is neutral and the
residual is qwenasr transcription content (frozen-reference pairing), and split/window are both dead
ends. Result pending.

## Caveats
- Trained eval only (synthesis-time BLEU is non-predictive — memory
  `feedback_synthesis_bleu_not_predictive`).
- ① vs ② mixes ASR source + decode method (win1/5-axis vs win3); the same-method 5-axis triple is
  the clean ASR isolation and agrees.
- The win1/win3 window was never isolated (see "Open: win1 vs win3"). The same-method 5-axis triple
  isolates ASR *within the 5-axis method* but leaves win1-vs-win3 confounded with ASR source, since ①
  is the only win1 run and the only old-asr run. Attacking it from the qwenasr side (qwenasr@win1) is
  cheaper than an old-asr@win3 decode and gives the same isolation.
- en→zh only. Standard prompt, [[checkpoint-evaluation]].

## Sources
- [[consensus-decoding]], [[2026-06-consensus-axis5-vs-futures200]], [[segale-alignment]],
  [[metricx]], [[dataset-conversion-pipeline]], [[scoreboard]], [[gigaspeech]],
  [[comet-vs-bleu-ranking]], [[infinisst-omni]], [[acl-6060]]
