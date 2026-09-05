# Research TODO

Updated: 2026-09-04

This is the working task list for the ambiguity-aware future-consensus
simultaneous translation project. Do not change the active 40K run's frozen
method while it is in progress; new sampler designs belong in separate pilot
runs with separate output roots.

## Current sprint (as of 2026-09-04 23:00 ET), in priority order

State: decode at 39,175 / 40,000 unique rows; tasks 2, 6, 9 and the gap-fill
are DONE; tasks 3, 4, 5, 7, 8, 10, 11 are finishing on resubmits (job IDs in
the run manifest). A 2-hour watcher in the Claude Code session resubmits
TIMEOUT tasks and will run the verifier at 40,000. Main tasks 0 and 1 were
released at ~10:40 ET and are regenerating rows already finished by task_12/13
(892 + 859 duplicate files so far).

### P0 — needs a decision from Haoling
- [ ] Cancel main-array tasks 0 and 1 (`scancel 10280652_0 10280652_1`). They
  cover no missing rows and burn 4 GPUs writing duplicate utterance IDs.
- [ ] Approve commit + push of the Simul-tst-COMMON eval chain change
  (`scripts/infer/run_infer_after_train_generic.sbatch`,
  `scripts/infer/eval_all_ckpts_simultst.sh`, `scripts/submit_ambiguity_40k.sh`),
  then `git pull --ff-only` on BABEL before the downstream chain is rebuilt.

### P0 — right after decode reaches 40,000
- [ ] Run the verifier over rows 0–39,999; expect duplicate IDs from task_00 /
  task_01. Build a one-JSON-per-utterance manifest (prefer task_12/13/14/15
  copies for rows 0–6667) before any downstream stage reads the decode root.
- [ ] Cancel the stale chain 10280664–10280673 and resubmit from SEGALE prepare
  through eval (ACL 6060 + Simul-tst-COMMON) with the committed launcher.
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

- [ ] Monitor BABEL run
  `ambiguity-q38-gemma-q36-fsetv2-prefixnorm-strict-40k-r1-20260831`.
- [ ] Preserve completed JSONs and keep total concurrent GPU use at or below 24.
- [x] Resolve the held-array coverage/dependency issue after current workers
  finish: the dual worker covers rows 0-4999, while held main-array task 1 also
  owns rows 5000-6667. (Gap-fill 10311095 completed rows 5000-6667 into
  task_14/15 on 2026-09-04. Dependency chain still to be rebuilt.)
- [ ] Verify the exact intended 40,000 utterance IDs, not only the file count.
- [ ] Repair the stale downstream Slurm dependencies without rerunning completed
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
