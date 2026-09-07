# Research TODO

Updated: 2026-09-05 23:01 ET

This is the working task list for the ambiguity-aware future-consensus
simultaneous translation project. Do not change the active 40K run's frozen
method while it is in progress; new sampler designs belong in separate pilot
runs with separate output roots.

## Current sprint (as of 2026-09-05 10:20 ET), in priority order

### P0 — restore a fair training-size comparison
- [x] Mark the 17,306-example checkpoint as uncontrolled/exploratory; do not use it as the
  primary comparison against 12,500-example baselines.
- [x] Sample exactly 12,500 rows from the same full survivor manifest with seed 42
  (job 10328626, verified 12,500/12,500).
- [ ] Complete matched training job 10328627 and automatic ACL 6060 + Simul-tst evaluation
  chain launched by job 10328628; replace the primary result table with matched scores.

State: decode COMPLETE, 40,000 / 40,000 unique utterances verified (0 missing,
0 unreadable). Main tasks 0 and 1 were cancelled 2026-09-05 10:05 ET after
writing 1,793 duplicate copies; those are excluded by the dedup symlink view
`<decode_root>-dedup` (verified 40,000 / 0 duplicates). The post-decode chain
was resubmitted from that view at 10:15 ET (jobs 10323222–10323232: SEGALE →
MetricX QE → length filter → convert → LoRA train → eval on ACL 6060 dev and
Simul-tst-COMMON). A 2-hour watcher in the Claude Code session follows it.

### P0 — needs a decision from Haoling
- [x] Cancel main-array tasks 0 and 1. (Cancelled by Haoling 2026-09-05 10:05 ET.)
- [x] Approve commit + push of the Simul-tst-COMMON eval chain change (commit 4df4927, pulled on BABEL)
  (`scripts/infer/run_infer_after_train_generic.sbatch`,
  `scripts/infer/eval_all_ckpts_simultst.sh`, `scripts/submit_ambiguity_40k.sh`),
  then `git pull --ff-only` on BABEL before the downstream chain is rebuilt.

- [ ] Decide the sampler-prefix window for the next pilot (2026-09-06 analysis in
  `wiki/experiments/2026-09-ambiguity-fsetv2-40k.md`, "What the sampler window really is"):
  A punctuation anchor (prefix = text after the last `. ! ?`), B anchor + previous full
  sentence, or C the implemented `until-closed` mode. Facts: units of `src_text_full` end
  with a comma 279/537 times, so the current unit anchor resets mid-sentence half the time;
  single-case A/B (job 10333550) confirms `--sentence-end-completion` removes the 碎了一地
  carry-over. Do not change the frozen 40k method; run as a separate pilot root.

- [ ] Simul-tst-COMMON repetition loops (found 2026-09-06, wiki experiment page section
  "Simul-tst repetition loops"): the TED intro jingle/applause makes the model loop `嘘！嘘！…`;
  matched-12,500 loses 4–10 BLEU per segment size to 1–5 runaway talks. Hibiki and EAST-even
  (same agent, same converter) never loop, every consensus checkpoint does → the cause is in our
  synthesized targets, not the pipeline. Haoling to discuss with the mentor before any change to
  `scripts/infer/infinisst_omni.py`; candidate data-side fixes: non-speech turns with empty
  targets in the converter, delta-style audit vs hibiki targets.

- [x] EAST low-only (multiplier 1–12) ablation cancelled 2026-09-07 at Haoling's request:
  61 % single-chunk examples. If revisited, use low→1–2 (and high→1–2) so streaming turns survive;
  converted data kept under `east_lowonly/`.
- [ ] Suffix-ICL v3 50-case pilot (`suffix-icl-v3-50cases-20260907-130701`): all four decode tasks
  of 10345238 died on the strict v3 response parser (Qwen3.8 on prefixes `I`, `A`, `Most sure and`
  and one 60-word prefix; deterministic under the seed). Cause: Qwen3.8 writes the heading
  `Contrast` instead of `Contrastive` on short prefixes. Fixed and rerun 2026-09-07 (generation
  10345647, report 10345648): 50/50 verified, published to the 8768 comparison viewer. v3 vs
  source-only boundary on the same 50 cases: char-BLEU −2.07 (16 up / 30 down), word LAAL −0.61
  (39 up / 10 down). Review 2026-09-07: v3 commits earlier but is not
  safer (1015 commits 进了 before the period; candidates per step 33.1 → 16.5). P1/P2 retry bugs
  fixed with tests. No v3 40k. v3 request switched to a vLLM JSON schema (commit dbfac56;
  probe 80/80 parsed, 8-case decode clean, 1015 no longer commits 进了 early). Next: 200 held-out
  cases, v2-boundary vs v3-JSON, XCOMET + step-level early-commit audit, candidate budget controlled
  (40 → 13.5 raw candidates per step is itself a variable).

- [ ] Probe input marks what was heard vs guessed (idea 2026-09-07): today the future is glued onto the
  observed English as if heard, so the translator commits content only the futures imply
  (`进了` in 1015). Change the probe prompt to `[HEARD] ... [POSSIBLE CONTINUATION] ...` and
  instruct it to translate only the heard part, using the continuation as context. Prompt-only;
  test on the same 50 cases against v3-JSON.
- [ ] Consensus voter floor: require a minimum absolute number of voters (or both samplers) in
  addition to the ratio, so fewer candidates (40 → 13.5 per step under the JSON schema) do not
  loosen strict consensus. Test with the item above.
- [ ] Schema field for contrastive candidates naming the reading each one resolves (strip before
  use); probe prompt reordering for prefix-cache hits; `--num-concurrent-cases` > 1 and parallel
  Gemma/Qwen sampler calls for throughput (see wiki notes 2026-09-07).

### P0 — right after decode reaches 40,000
- [x] Run the verifier over rows 0–39,999 and build the one-JSON-per-utterance view.
  (Raw root: 40,000 unique, 0 missing, 1,793 duplicates from task_00/01. Dedup
  symlink view `<decode_root>-dedup` built by `external_runner/dedupe_decode_root.py`,
  verified 40,000 / 0 duplicates, 2026-09-05.)
- [x] Cancel the stale chain 10280664–10280673 and resubmit from SEGALE prepare
  through eval. (Resubmitted 2026-09-05 10:15 ET via `scripts/resubmit_ambiguity_40k_post.sh`,
  jobs 10323222–10323232; 2-hour watcher active in the Claude Code session.)
- [ ] Record survivor counts after MetricX QE ≤ 3.0 and after the 0.7–1.5
  length filter; report BLEU / LAAL / XCOMET vs top5-axis5 and hibiki on both
  test sets.

### P1 — mentor-facing, CPU only, can start now
- [ ] Verbose-log profiles for the 10 listed cases (4 failures, 6 successes):
  decisive chunk, raw candidates with filter verdicts, selected futures by
  model/mode, per-future probe distributions at each consensus step, the
  intersection where agreement broke, horizon/commit lines, and a one-line
  mechanism label (probe bias / prefix-reset context loss / missing
  dialogue-boundary future / genuine divergence → READ). Done so far:
  `AUD0000000003_1059` (success, verb-sense split) and `AUD0000000003_1011`
  (failure, 36/36 futures agree on 它们). Template: local scratch logs from
  2026-09-04.
- [ ] Per-profile Gemma accounting: how many Gemma candidates were dropped at
  the decisive step and why, and whether a Gemma future carried the correct
  reading (in `_1059` the only "in thought" future was Gemma's and was dropped
  as `too_short`). Feeds the sampler ablation below.
- [ ] Put the viewer in the cloud: inline `data/review.json` into `index.html`
  (~7.5 MB, no audio, no raw logs), add the 3 + 3 landing section with
  `#case=` links, publish as a private Artifact, hand the HTTPS link to Siqi.
  Audio requires a separate permission decision (GigaSpeech clips) and either
  the BABEL server + tunnel or an authenticated host.
- [ ] Reply to Siqi's 2026-09-03 Slack message (GPT/Gemini baseline outputs and
  the rate-adaptive manifest under `siqiouya/results/simuls2s/share/`).

### P1 — sampler capacity (the Gemma question)
- [ ] Hypothesis: Gemma-4-E2B is too small — 70% keep rate vs 90% for Qwen3.8,
  many `too_short` drops, occasional hallucinated or off-topic continuations.
  Test a larger instruction-tuned second sampler on a dedicated third GPU
  (candidates: a mid-size Gemma-4 checkpoint or a second Qwen3.8 variant),
  after an L40S memory-fit + vLLM smoke test. Keep the Qwen3.6 translator
  isolated and benchmark sampler throughput vs translator queue.
- [ ] Build the ambiguity-stratified eval set (100 → 500) from the finished
  decode; then run the three matched arms (Qwen-only 20, Qwen-only 40,
  Qwen + Gemma 20+20) and the larger-sampler arm on it. Details and decision
  rule in the P1 section below.

### P2 — after the frozen run is scored
- [ ] Fragmentation ablation: merge deltas shorter than 2 characters into the
  neighbouring delta on the finished 40k decode, identical convert / train /
  eval, compare to the unmodified run (same design as the period-fix ablation).
- [ ] Method fixes from the P2 section below (context across prefix resets,
  gender/antecedent constraint, dialogue-boundary futures, direct early-commit
  evaluation).

---

## P0: Finish and verify the active 40K experiment

- [x] Monitor BABEL run
  `ambiguity-q38-gemma-q36-fsetv2-prefixnorm-strict-40k-r1-20260831` decode (complete 2026-09-05 06:53 ET).
- [ ] Preserve completed JSONs and keep total concurrent GPU use at or below 24.
- [x] Resolve the held-array coverage/dependency issue after current workers
  finish: the dual worker covers rows 0-4999, while held main-array task 1 also
  owns rows 5000-6667. (Gap-fill 10311095 completed rows 5000-6667 into
  task_14/15 on 2026-09-04. Dependency chain still to be rebuilt.)
- [x] Verify the exact intended 40,000 utterance IDs, not only the file count.
- [x] Repair the stale downstream Slurm dependencies without rerunning completed
  decode rows.
- [ ] Complete 24-shard SEGALE alignment and verify all shard sentinels.
- [ ] Complete 24-shard MetricX QE, retain cases with maximum sentence QE <= 3.0,
  and record the survivor count.
- [ ] Apply reference-length ratio filter 0.7-1.5 and record the survivor count.
- [ ] Convert to training JSONL, LoRA-train Qwen3-Omni, export the checkpoint,
  and report BLEU, LAAL/latency, and XCOMET.

## P1: Decide whether to replace Gemma or remove it

Hypothesis: `gemma-4-E2B-it` may be too weak for coordinated ambiguity-focused
future generation. In the 100-case audit its keep rates were 70.0% plausible
and 72.0% contrastive, versus 90.4% and 88.1% for Qwen3.8. This is evidence of
lower filter pass rate, but it does **not** yet prove that Gemma hurts consensus:
its model diversity may still contribute useful alternatives.

- [ ] Build one fixed, ambiguity-stratified evaluation set containing lexical,
  syntactic, referential/gender, negation, attachment, and speaker-boundary
  cases. Start with 100 cases, then confirm on at least 500.
- [ ] Run a matched-compute ablation: Qwen3.8 only, 20 candidates total.
- [ ] Run a matched-candidate ablation: Qwen3.8 only, 40 candidates total.
- [ ] Run the current control: Qwen3.8 + Gemma, 20 candidates per model.
- [ ] Test a larger instruction-tuned second sampler on a dedicated third GPU.
  Select the checkpoint only after confirming L40S memory fit and vLLM support.
- [ ] If using three GPUs, keep the Qwen3.6 translator isolated and benchmark
  whether sampler throughput or the translator request queue is the bottleneck.
- [ ] Keep the generic `future_set_v2_two_groups` prompt and the same filtering
  rules across ablations so the model comparison is valid.
- [ ] Use at least three seeds or otherwise quantify sampling variance.
- [ ] Compare parsed candidate count, filter reason distribution, exact/near
  duplication, semantic diversity, ambiguity coverage, model contribution to
  selected consensus futures, GPU memory, tokens/s, and rows/hour.
- [ ] Compare downstream early-commit failures, BLEU, XCOMET, and LAAL. Do not
  select a model using BLEU or raw keep rate alone.
- [ ] Inspect whether Qwen-only candidates collapse to one model-specific bias.
  A larger second model is preferable only if it adds grounded contrastive
  futures rather than merely more fluent candidates.
- [ ] Do not describe Gemma as "4B" in a paper or report until the exact
  checkpoint's parameter convention is verified; use its checkpoint name for
  now.

### Decision rule

- Remove Gemma if Qwen-only matched-candidate decoding preserves or improves
  ambiguity coverage and early-commit safety while materially reducing cost.
- Replace Gemma if a larger second model adds complementary, grounded futures
  and improves early-commit safety enough to justify the third GPU.
- Keep Gemma if its lower-pass candidates still provide unique ambiguity
  coverage that disappears in Qwen-only runs.

## P1: Curate ambiguity examples for the mentor

Use the trajectory viewer and show the exact source prefix, selected futures,
READ/WRITE decision, committed Chinese delta, and the later disambiguating
source. Separate successful waiting behavior from failures.

### Confirmed failure examples

- [x] Case 9, `AUD0000000003_1011`: at `They were so huge that the`, the decoder
  commits `它们如此巨大，以至于`; later context refers to people/giants, so the
  pronoun should be `他们`. This demonstrates sampler/referent bias.
- [ ] Case 71, `AUD0000000003_1125`: a normalized-prefix reset loses antecedent
  and gender context, causing irreversible male `他` before female resolution.
- [ ] Case 83, `AUD0000000003_1152`: after `Here I must remain no`, the next
  normalized prefix resets and loses negation, producing the opposite meaning.
- [ ] Case 64, `AUD0000000003_111`: punctuation-poor ASR hides a speaker change;
  futures omit the dialogue-boundary alternative and attach speech to the wrong
  speaker.

### Strong success examples

- [ ] Case 1, `AUD0000000003_0`: READs through `introductions are inevitably`
  and commits only stable `而这些介绍` before unresolved adjectives.
- [ ] Case 25, `AUD0000000003_1038`: commits a safe Chinese alternative frame
  while waiting for `buried in the snow`.
- [ ] Case 35, `AUD0000000003_106`: `No, I won't` commits only `不`, postponing
  the unresolved complement.
- [ ] Case 58, `AUD0000000003_1100`: `Before her stood` commits only
  `在她面前`, leaving the object unresolved.
- [ ] Case 90, `AUD0000000003_1171`: `This so enraged the` commits only `这使`
  and waits for `king`.
- [ ] Case 97, `AUD0000000003_1182`: `When he heard a` commits only `当他听到`
  and waits for the sound type.

### Mentor deliverable

- [ ] Create a short landing section containing 3 successful and 3 failed
  cases rather than asking the mentor to browse all 100 first.
- [ ] Add a one-sentence explanation of what uncertainty remained at each
  highlighted step.
- [ ] Label errors as `early commitment`, `boundary ambiguity`, or `MT semantic`
  so final translation quality is not confused with timing safety.
- [ ] State that the current 100 cases are contiguous from one recording and
  are illustrative, not a representative benchmark.

## P1: Publish the trajectory viewer for sharing

Current local-only URL: `http://127.0.0.1:8766/`.

- [ ] Decide whether the mentor site may be public or must require
  authentication. Confirm permission before publishing GigaSpeech audio or raw
  model logs.
- [ ] Build a sanitized deployment directory containing only `index.html`,
  `styles.css`, `app.js`, `data/review.json`, and approved audio.
- [ ] Exclude raw verbose logs, credentials, model paths, scheduler logs, and
  unrelated research outputs from the hosted artifact.
- [ ] Add the curated mentor examples to the initial page or provide direct
  case links using `#case=<utterance-id>`.
- [ ] Deploy to a stable HTTPS host and record the deployment source/version.
- [ ] Test the URL in a signed-out/incognito browser, including case search,
  direct case links, mobile layout, and audio playback.
- [ ] Send the mentor the HTTPS URL plus the six recommended case links.
- [ ] Keep the BABEL/group-data bundle as the archival source; do not treat the
  public website as the canonical dataset.

## P2: Improve the method after the frozen 40K run

- [ ] Carry unresolved source and target context across prefix-normalization
  boundaries; never evaluate a new `and` or short clause fragment without an
  unresolved antecedent or negation from the previous clause.
- [ ] Add an explicit constraint for Chinese gendered pronouns: if futures
  disagree on gender, force READ or use safe neutral/name-based wording.
- [ ] Add same-speaker, speaker-switch, and reported-speech alternatives for
  punctuation-poor ASR.
- [ ] Add direct early-commit evaluation using bilingual human annotation or an
  oracle full-source translation alignment. BLEU and XCOMET alone cannot
  detect when an irreversible decision happened.
- [ ] Build a stratified review set across recordings and ambiguity types before
  reporting an overall early-commit failure rate.

## Done log

- 2026-09-04: verified 21 steps where waiting on the futures was decisive
  (`data_synthesis/reports/future_consensus_success_cases_2026-09-04.md`); aggregate: READ rate
  ~56% regardless of prefix ending, real continuation appears among futures in
  23% of steps. Verbose profiles for `_1059` and `_1011` pulled from
  production task_00. Simul-tst-COMMON eval added to the post-train launcher
  (uncommitted). Timed-out decode tasks 2, 4, 5, 6, 7, 8, 10, 11 resubmitted
  with manifest records; gap-fill 5000-6667 completed.

## P2: Repository follow-up

- [ ] Monitor Open-LiveTranslate PR #34:
  `https://github.com/LeiLiLab/Open-LiveTranslate/pull/34`.
- [ ] Address maintainer review on branch
  `codex/future-consensus-external-runner`; do not push directly to `main`.
- [ ] Run the official `future_consensus` data test target inside
  `$OLT_VENV_ROOT/olt-main` before the next PR update.
- [ ] Keep model weights, datasets, SIF images, outputs, caches, and credentials
  outside Git.

See `CLAUDE_CODE_HANDOFF_2026-09-03.md` for full operational context.
