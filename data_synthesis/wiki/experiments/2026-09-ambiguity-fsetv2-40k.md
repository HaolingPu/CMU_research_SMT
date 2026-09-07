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

**Status: uncontrolled exploratory result. Do not use this checkpoint as the fair baseline
comparison.** It trained on 17,306 examples while the established baselines trained on 12,500.
A matched 12,500-example rerun (`-n12500-seed42`) was submitted on 2026-09-05 as jobs
10328626–10328628 and must replace the numbers below in any primary comparison.

**Exploratory answer.** BLEU rises by 6–8 on [[acl-6060]] and by ~11 on
[[simul-tst-common]] at every latency ≥ 1920 ms, passing the ref-based hibiki system on both
sets. XCOMET falls by 0.01–0.04. These numbers are not a controlled result: training-set size
and the probe/sampler swap are both unresolved confounds.

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
| **matched 12,500** (seed 42, ckpt `v2-20260906-174211-hf`) | 40.7 / 47.4 / **48.5** / 48.4 | .767 / .791 / **.803** / .799 | 1414 / 2105 / 2648 / 3167 |
| `top5-axis5` flagship | 34.9 / 39.6 / 40.1 / 40.1 | .787 / .808 / .812 / **.817** | 1461 / 2176 / 2745 / 3107 |
| hibiki (ref-based) | – / – / – / 46.8 | .780 / .812 / .814 / .820 | – / – / – / 3326 |
| EAST-even | – / – / – / 46.8 | – / – / – / .789 | – / – / – / 3533 |

chrF this run: 36.1 / 39.8 / 40.5 / 40.7. chrF matched 12,500: 36.5 / 40.4 / 41.3 / 41.6.
Matched-count control (ACL eval job 10333149, 2026-09-06): training on 12,500 of the same
17,306 survivors matches or beats the 17,306-row run at every segment size, so the ACL gain over
the `top5-axis5` flagship is not a training-size artifact. Simul-tst-COMMON for this checkpoint:
pending (infer 10333150 + repair 10333556, eval 10333151).

## Results — Simul-tst-COMMON, monotonic refs (seg 960 / 1920 / 2880 / 3840)

| system | BLEU | XCOMET | LongYAAL CU ms |
|---|---|---|---|
| **this run** | 20.8† / 42.8 / 45.0 / **45.9** | .769† / .840 / .857 / .860 | 15495† / 2078 / 2442 / 2884 |
| **matched 12,500** (seed 42, eval 10333151) | 23.9‡ / 31.8‡ / 39.9‡ / 42.2‡ | .800 / .828 / .830 / .846 | 9768‡ / 16170‡ / 29340‡ / 4262‡ |
| `top5-axis5` flagship | 27.5 / 32.1 / 34.1 / 34.2 | .831 / .859 / .867 / **.872** | 3535 / 1543 / 2409 / 2855 |
| hibiki (ref-based) | 38.3 / 40.4 / 40.8 / 41.1 | .838 / .861 / .866 / .869 | 1282 / 1849 / 2429 / 2870 |
| EAST-even | 40.1 / 43.7 / 44.2 / 43.6 | .765 / .817 / .837 / .846 | 1107 / 1900 / 2620 / 3243 |

chrF this run: 28.3 / 36.7 / 38.7 / 39.0. † seg960 is degenerate (see hygiene) and should be
excluded from any comparison. ‡ every matched-12,500 segment size carries 1–5 runaway talks
(repetition loops on non-speech audio, see "Simul-tst repetition loops" below); with the loops
stripped the same outputs score 37.9 / 41.8 / 43.5 / 44.4 and with the runaway talks dropped
37.1 / 42.2 / 45.3 / 46.1, i.e. on par with the 17,306-row run.

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

## Profiling (2026-09-06): sentence-boundary carry-over and invented filler

Trigger case: `AUD0000000003_1015` — "And the fork flew into a dozen pieces." was committed as
叉子飞出去，摔成了十几块**，** (trailing comma, weakest vote of the case: 6/15 futures top-1,
min p 0.16), the sampler prefix then reset to the next sentence, and three chunks later all 40
futures agreed on 碎了一地。 (p̄ 0.38 → 1.0) before 这巨人… began. "Shattered all over the ground"
is not in the source. Mechanism = the case-71 reset problem in a new guise: once the committed
Chinese is left mid-sentence, the probe's continuation of that clause is independent of the
(now off-topic) futures, so unanimity carries no safety information.

Scan of all 100 cases (viewer trace files):
- 169 WRITE chunks coincide with an English sentence end: 43 % close the Chinese sentence,
  49 % end without punctuation (translation lagging, normal), **9 % (15) end on a comma**.
  Comma votes are usually strong (median top-1 share 0.92); case 1015 is the only weak one.
- 73 unique "carry-over" events (Chinese ended on a comma at a reset, next WRITE closed the old
  sentence before starting the new one). LLM annotation (sonnet, 1 annotator + 2 independent
  second reads on every 'ungrounded' label): **62 grounded** (remaining content of the ended
  sentence), **8 next-sentence** (punctuation misplaced only), **3 ungrounded**, 0 unclear.
- The 3 ungrounded, all confirmed 2/2: `_1015` step 8 碎了一地 (invented location detail);
  `_1169` step 25 全部招认 after 玛丽亚毫不隐瞒 for "Maria denied nothing" (paraphrase doubled);
  `_1186` step 23 陷入极度焦虑之中 after 担心…会被弄糟 for "was in agony lest…" (intensifying
  restatement). Comma votes there were 6/15, 35/35, 16/20 — the vote strength does not
  discriminate; leaving the sentence open does.

Takeaways: (1) invented content at sentence boundaries is rare in this run (3/100 cases,
1 with a genuinely new fact) but it is a distinct failure class from early commitment;
(2) rule candidates: forbid a commit ending in sentence-internal punctuation once the source
sentence has ended, or route a finished-but-unclosed sentence through the final-completion
path; and the reset-aware future window. Tooling: the trajectory viewer now renders the full
next-token trace per step (`data/consensus/<utt>.json`, commit `db57a69`).

## Single-case A/B of the boundary fixes (2026-09-06, job 10333550, `AUD0000000003_1015`)

Decoder commit `3077644` adds three flags; `experimental/run_single_case_ab.sbatch` decodes one
TSV row under several flag sets with the production samplers/probe loaded once, all variants
sharing `--targeted-sampler-seed 1015` (seed + crc32 of prefix|committed|model, so futures are
identical across variants until the trajectories diverge). Outputs:
`/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_pilots/single_ab_1015_20260906-1110/<variant>/task_00/`.

| variant | flags | step-4 commit (source `pieces.`) | 碎了一地 | char-BLEU | LAAL |
|---|---|---|---|---|---|
| baseline | — | 散成十几块 (no punctuation; `碎片。` lands at step 6) | no | 59.3 | 6.12 |
| joinfix | `--future-join-mode sentence-aware` | 散成十几块**，** | **yes, step 11** | 24.2 | 9.85 |
| joinfix_sentend | + `--sentence-end-completion` | 成了十几块碎片**。** (`[SentenceEnd]`) | no | 57.8 | 6.54 |
| full | + `--future-source-window-mode until-closed` | 成了十几块碎片。 | no | **66.0** | 6.16 |

Reading: (1) the carry-over reproduces exactly when the source period is committed with a
Chinese comma (joinfix) and disappears when the sentence-end completion closes it with 。;
(2) the seed makes the futures reproducible (step-4 candidate lists are byte-identical between
baseline and joinfix) but the probe's logprobs are not bit-stable across runs, so a 6/15-style
vote can flip a trailing comma between otherwise identical variants — single-case deltas in
BLEU are noise, the structural effect on the boundary is the signal; (3) `until-closed`
widened the sampler window to 2 units for chunks 10–15 because unit 1 (`...had been,`) left
the Chinese open, but at chunk 12 it still dropped `This giant was...` because the mode caps at
2 units. Viewer bundle with the four logs: `/private/tmp/trajectory-viewer-ab1015`
(laptop, port 8767).

### What the sampler window really is (100-case measurement)

`--future-source-window-chunks N` counts **units of `src_text_full`**, not ASR chunks, and the
sampler prefix always starts at the beginning of the current unit (`build_source_observed_recent_units`).
So an anchor at the unit start already exists. The problem is the units: over the 100 viewer
cases, `src_text_full` has 537 units and only 230 end with `. ! ?`; 279 end with a comma and 28
with no punctuation. Half of all sampler resets therefore happen mid-sentence, and at the first
step of a new unit the sampler sees a median of 2 words (262/437 unit starts show ≤2 words).

| step type (chunk position) | steps with futures | WRITE rate |
|---|---|---|
| starts a true sentence (after `. ! ?`) | 310 | 34.8 % |
| starts a comma-fragment unit | 251 | 53.4 % |
| mid-unit | 1857 | 48.4 % |

| sampler-prefix length | steps | Gemma restart-as-new-sentence | Qwen |
|---|---|---|---|
| 1–2 words | 335 | 1.6 % | 0.5 % |
| 3–4 words | 341 | 0.7 % | 0.1 % |
| 5–8 words | 685 | 4.7 % | 0.2 % |
| 9+ words | 1057 | 5.8 % | 0.2 % |

Gemma's sentence restarts grow with prefix length, so widening the sampler window does not
buy fewer restarts (the sentence-aware join handles those on the probe side). The 73
carry-over WRITE steps sit mostly a few chunks into the new unit (46 mid-unit, 5 fragment
starts, 4 sentence starts), and 21/73 had a sampler prefix of ≤2 words.

Design options for the sampler prefix (decision pending, see RESEARCH_TODO):
- **A. punctuation anchor** — prefix = observed text after the last `. ! ?`; ignores the
  comma-fragment units, no 2-unit cap, no LLM call. Cheapest fix for the 262 mid-sentence resets.
- **B. A + previous full sentence** — always prepend the previous sentence (or only when the
  current-sentence prefix has < K words). Restores the antecedent that the case-71 gender
  error lost; costs a longer prompt and slightly more Gemma restarts.
- **C. until-closed** (implemented, `full` variant) — target-state driven, needs
  `--sentence-end-completion` to be meaningful, capped at 2 units.
June history: win3 (3 units, fixed) was confounded with the ASR change and never isolated
([[2026-06-qwenasr-asr-regression-periodfix]]), so a wider *fixed* window has no clean
evidence either way.

## Simul-tst repetition loops (2026-09-06): why Simul-tst is bad while ACL is good

Every MuST-C tst-COMMON wav opens with the same ~12 s TED intro jingle plus applause
(the per-0.96 s RMS envelope of the first 15 s is identical, up to gain, across all 27 talks;
the first yaml segment simply absorbs it). Talks also contain applause after "thank you",
laughter and embedded video clips. The GigaSpeech training clips contain none of this, and
nothing at inference brakes repetition (`--repetition-penalty` default 1.0, temp 0.6, top-p 0.95,
every assistant turn appended to the chat history). On non-speech audio the model emits an
interjection, the interjection enters the history, and the next chunk of non-speech continues it:

| loop unit (verbatim) | where it starts | example |
|---|---|---|
| `嘘！嘘！嘘！…` | first writes at 2.9–9.6 s, inside the jingle | matched seg960 talks 0, 1, 7, 8, 10 (63–100 % of the talk's output) |
| `谢谢。谢谢。…` | closing applause after 谢谢 | talk 0 (ted_1096) at ~205 s in both runs, 47 % of the talk at seg1920 |
| `呼气，吸气，…`, `哦，我的天哪，…`, `是的，是的，…` | sound effects / video clip / applause | matched seg3840 talks 14, 22; full seg960 talk 5 |
| `（掌声）`, `（音乐）`, `（音效）` | short, self-limiting | the 17,306-row run labels non-speech more often instead of looping |

Loop census (a unit of 1–14 chars repeated 8+ times), talks with any loop / share of all output
characters inside loops: matched 25/27 (35.8 %), 11/27 (27.6 %), 3/27 (12.3 %), 5/27 (11.2 %) for
seg 960/1920/2880/3840; 17,306-row run 22/27 (17.0 %), 5/27 (2.1 %), 2/27 (0.7 %), 2/27 (0.8 %).
Runaway talks (>50 % of the talk inside loops): matched 5 / 2 / 1 / 2, 17,306-row 3 / 0 / 0 / 0.
seg960 loops most because the jingle is 12 chunks long before any speech arrives.

Cost of the loops (char-BLEU recomputed from `instances.resegmented.jsonl`, reproduces the
official numbers exactly; mean per-segment emission delay in ms):

| run, seg | BLEU official | loops stripped | runaway talks dropped | emission all → w/o runaway |
|---|---|---|---|---|
| matched 960 | 23.9 | 37.9 | 37.1 | 12494 → 3961 |
| matched 1920 | 31.8 | 41.8 | 42.2 | 18887 → 5078 |
| matched 2880 | 39.9 | 43.5 | 45.3 | 31877 → 5080 |
| matched 3840 | 42.2 | 44.4 | 46.1 | 6780 → 5515 |
| 17,306 1920 | 42.8 | 43.9 | 42.8 | 4672 → 4672 |
| 17,306 3840 | 45.9 | 46.2 | 45.9 | 5419 → 5419 |

One runaway talk (8,000–17,000 characters of `嘘！`) is enough to sink corpus BLEU by 4–10 points
and to push LongLAAL into the tens of seconds, because mwerSegmenter aligns the garbage across
many reference segments. On clean speech the matched checkpoint is not worse than the
17,306-row one; which talks spiral is sampling luck on the jingle. ACL 6060 dev has no jingle,
no applause and no clips, so neither checkpoint loops there (rep-4gram ≤ 0.22 on all 5 talks).

Independent check (three-lens workflow, 2026-09-06 evening, scripts in scratchpad `wf/`):
- Onset: on the matched checkpoint the first write lands inside the 0–15.4 s jingle window on
  26/26/23/22 of 27 talks (seg 960/1920/2880/3840) and is an interjection or a sound description
  (`嘘！`, `嘶嘶声。`, `（掌声）`, `（音乐）`), never translation; the first reference-matching write
  comes at 17–19 s in every run. ACL: 40/40 first writes are real translation at 2.9–3.8 s.
  The 17,306-row model even narrates the trigger: `*音效：开场音效，包含呼啸声、电子音效和掌声，随后是掌声。*`.
- Persistence is the chat history, not the audio: 20/25 jingle-`嘘！` talks recover once speech
  starts; the 5 runaway talks are exactly those with a max-new-tokens (30-char) stutter turn
  among the first 15 turns (5/7 with such a turn ran away, 0/20 without); 96.6 % of loop
  characters sit in turns ≥ 25 chars. History trimming (60→30 turns) does not break a loop.
- Mid-talk onsets are all non-speech: closing applause over 20–50 s yaml segments
  (`Come back.` 223–245 s in ted_1096 followed by a Siemens sponsor ad), the Double Rainbow
  clip `Oh, my God!` ×4 in ted_1371, an embedded video in ted_1359. Only two matched loops
  (<2 % of loop chars) start in ordinary speech.
- Scoring is sound: BLEU reproduced to 4 decimals for all 16 runs, references/yaml/order all
  match. One uniform defect: hypotheses are NFKC-normalized but references are not (9.5 % of
  reference tokens are full-width punctuation); fixing it adds +4–8 BLEU to every row and does
  not change any gap. Energy gating is ruled out: the jingle is 1.1–8.6× louder than speech.
- Attribution: on talks clean in both runs the matched checkpoint is ≥ the 17,306-row one
  (seg2880 46.7 vs 46.1, seg3840 47.0 vs 46.1); at seg960 the 17,306-row run is the worse one
  (20.8 vs 23.9). Evidence that matched is more loop-prone is weak and post hoc
  (pooled Fisher p = 0.36, one seed per run).
- Fix order: A history-level loop brake in `infinisst_omni.py` (periodic-turn, interjection-only,
  and 2-turn-repeat detectors → write `''` to history, return READ; 23 % of training turns are
  empty so this is in-distribution) first, with `--presence-penalty 0.4` as a zero-code parallel
  arm; then C a Silero-VAD generation gate (audio kept in history); D converter-level non-speech
  turns with empty targets needs a retrain; `--repetition-penalty` > 1 last (penalizes the whole
  history). Success metric: runaway talks → 0, LongLAAL CU back to 2–3 s, ACL BLEU within ±0.3.
Is it the shared inference agent? No. Loop census over every checkpoint's saved Simul-tst
outputs on BABEL (same `infinisst_omni.py`, same converter; share of output characters inside a
loop / runaway talks, seg 960 / 1920 / 2880 / 3840):

| checkpoint (training targets) | loop share % | runaway talks |
|---|---|---|
| hibiki (word-aligned reference) | 0.9 / 0.0 / 0.0 / 0.0 | 0 / 0 / 0 / 0 |
| EAST-even (reference, even split) | 0.0 / 0.0 / 0.0 / 0.0 | 0 / 0 / 0 / 0 |
| consensus top5-axis5 (flagship) | 11.1 / 2.5 / 0.0 / 0.0 | 1 / 1 / 0 / 0 |
| consensus PA-40k | 19.8 / 7.2 / 2.6 / 3.3 | 3 / 2 / 1 / 0 |
| consensus anchor40k | 48.6 / 17.2 / 8.3 / 5.7 | 2 / 3 / 1 / 2 |
| consensus bestof4refsel / bestof5refsel | 15.5 / 17.6 / 0.0 / 0.0 and 24.8 / 0.1 / 0.3 / 0.3 | 2 / 1 / 0 / 0 and 2 / 0 / 0 / 0 |
| ambiguity 17,306 (this run) | 17.0 / 2.1 / 0.7 / 0.8 | 3 / 0 / 0 / 0 |
| ambiguity matched 12,500 | 35.8 / 27.6 / 12.3 / 11.2 | 5 / 2 / 1 / 2 |

Hibiki at seg960 emits `嘘！` in 23 talks but only ~2 per talk and recovers at once. Every
consensus-family checkpoint loops; the two reference-target baselines never run away. The
susceptibility is therefore a property of the synthesized training targets, not of the
train-infer-eval pipeline, and a decoding-time brake would help only our systems. In the
100-case bundle 6.1 % of chunks have empty ASR text (in-clip silence) and 26/156 of those still
commit a (grounded, lagged) delta; whether that or the delta style drives the susceptibility is
not established (see [[2026-09-ambiguity-12500-tst-failure-audit]] for the read-only audit).
Decision pending with the mentor: report as-is with the loop diagnostics, and treat the fix as
a data-side question (non-speech turns with empty targets, delta style) rather than an agent change.
Outputs: `<ckpt>/evaluation/simul_tst_common/en-zh/seg<N>/instances.log`; local copies and the
analysis scripts in the session scratchpad `tst_bundle/`, `tst_audio_rms.json`.
Fix candidates (none applied yet; the inference agent is `scripts/infer/infinisst_omni.py`):
history-level loop brake (drop a turn that repeats the previous turns or is interjection-only,
return READ), `--repetition-penalty` > 1, an energy/VAD gate for non-speech chunks, and
training turns with non-speech audio and empty targets.

## Next

- **In progress:** matched 12,500-instance rerun, seed 42, sampled from the same 17,306-row
  full manifest. Resample job 10328626 completed and verified exactly 12,500 rows; training
  job 10328627 and automatic ACL + Simul-tst evaluation launcher 10328628 are queued.
- Per-sentence COMET diff vs flagship on tst seg3840 (July forensics recipe).
- Inspect tst seg960 outputs; consider an inference-time repetition brake.
- Sampler ablations on an ambiguity-stratified set (Qwen-only 20/40, + larger Gemma-4 12B/31B):
  metric = fraction of ambiguity steps where a sampler produces ≥1 cue-carrying future.
- Sampler-window redesign: pick between options A/B/C above (punctuation anchor, + previous
  sentence, until-closed) and test on a stratified pilot, not on the frozen 40k method.

## Related
- [[ambiguity-future-set]], [[consensus-decoding]], [[future-sampling]], [[scoreboard]],
  [[comet-vs-bleu-ranking]], [[simul-tst-common]], [[acl-6060]], [[latency-quality-tradeoff]],
  [[2026-06-consensus-axis5-vs-futures200]], [[2026-07-consensus-register-forensics]],
  [[2026-07-present-propose-gate]], [[2026-07-anchor-smoke500-sweep]], [[synthesis-pipeline]],
  [[qwen3-omni]], [[babel-cluster]].
