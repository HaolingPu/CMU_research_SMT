# Same-50-case suffix-ICL rerun and paired viewer

## Scope

Requested 2026-09-07: preserve the completed cases and rerun the same 50 with
the revised prompt, including the nested-relative-clause and reduced-passive
ambiguity examples. Do not run 40K or train a new model yet.

- Run: `suffix-icl-v3-50cases-20260907-130701`.
- Babel manifest: `/home/haolingp/slurm_runs/suffix-icl-v3-50cases-20260907-130701/run_manifest.json`.
- Generation: `10345238`, array `0-3%4`, two GPUs per worker, maximum eight GPUs.
- CPU validation/report/bundle: `10345239`, depends `afterok:10345238`.
- Existing Simul-MuST-C production generation stays at 16 GPUs; later stages
  do not overlap that generation and request at most 16. Combined maximum 24.
- `preempt/preempt_qos`, eight-hour limit, requeue; CPU uses `preempt_cpu_qos`.
- Confirmed prior/precautionary exclusions are copied from the current job;
  `babel-n5-24` was also excluded because live node state reported a prolog error.
- Remote `feature/home-checkout` was pulled `--ff-only` before staging.
  The runtime is an explicit immutable copy of local uncommitted files, with
  per-file hashes in the manifest, not a claim that HEAD contains these edits.

## Paired treatment

Old comparison arm: `source_only_boundary` from
`source-only-sentence-boundary-50cases-20260907-023037`.
Its complete 50-case output is reused, not regenerated.

Input SHA256:
`d0ba56e944c14f340a61146acf0658e3ae8548b5ee0f95c60ea6a1e4c4bae68d`.
Exactly the old ordered selection, seed 42, sampler seed 1015.

New arm `suffix_icl_v3` adds
`--targeted-prompt-version future_set_v3_suffix_icl` to the unchanged old flags:

```text
--targeted-sampler-context source-only
--future-source-window-mode sentence-anchor
--sentence-end-completion
--sentence-end-boundary-mode conservative
--sentence-end-punctuation match-source
```

The treatment is the v3 prompt plus its variable-size grouped response parser.
The prompt has five ambiguity demonstrations and prioritizes literal suffix
fit, continuation of open grammatical structure, and realistic alternatives.
It allows fewer than 20 candidates per sampler (up to 10 per group) instead of
padding bad candidates. Strict malformed-response detection is not silent
abstention. No Gemma removal, model replacement, new filter, consensus change,
or training change is included.

Models remain Gemma `gemma-4-E2B-it`, Qwen sampler `Qwen3.8-27B-FP8`, and
translator `Qwen3.6-35B-A3B-FP8`. Each worker colocates both samplers on one GPU
and serves the translator on the other. Same seed does not make vLLM probes
bitwise deterministic. Treat differences as pilot diagnostics, not proof of
downstream training improvement.

Full prompt: [preview](2026-09-future-suffix-icl-prompt-preview.md).

## Preservation and website

Durable old archive (relative to the repository):
`data_synthesis/outputs/trajectory_reviews/archives/source-only-sentence-boundary-50cases-20260907-023037`.
All 416 copied files were SHA256-verified; the sibling `.sha256.json` is the
hash index. It includes baseline and source-only output JSONs, verbose logs,
traces, metrics, selection, audio and the original website assets.

Served bundle: `/private/tmp/trajectory-viewer-source-only50-20260907`.
Comparison: `http://127.0.0.1:8768/experiments.html#case=AUD0000000003_1015`.
All three arms are listed in selectors. Old source-only is the default left
arm and new v3 is the default right arm. The 8766/8767 viewers are unchanged.

Until the new run completes, the right side explicitly waits. It must not show
zero metrics or copy old results under a new label. Every case and regression
remains visible. Input/ref/chunk mismatches stop the comparison.

When report 10345239 is complete, retrieve its `review/` (under the manifest's
run directory) to a durable local directory, then use
`data_synthesis/tools/trajectory_viewer/manage_experiments.py publish` with the
served bundle as `--root`. The gate verifies the run, input checksum, exact
50-case coverage, recorded v3 settings, matching input/reference/chunks and
raw prediction identity. No old raw/data file may be changed during publication.

## Verification at submission

49 future-sampling unit tests and 16 viewer/publish tests passed. Browser tests
passed at desktop and mobile widths: pending is not zero, real historical
regressions remain visible, same-case navigation survives switching tabs,
exact prefix whitespace is preserved, HTML-like prefixes render literally,
and mismatched source inputs are rejected. Browser fixtures are not real
results and are never published.

The first worker started on `babel-s5-32`; Qwen reported ready after 95 seconds,
Gemma after another 62 seconds in the harness health sequence, and the
translator was already ready. About 158 seconds elapsed before decoding.
Real verbose logs began growing with v3 input auditing. This is a startup
observation, not a completed 50-case result.

By the end of this setup, logical workers 0 and 1 were running and workers
2/3 were Priority pending. The first two utterance JSONs existed; row 0
(`AUD0000000003_1015`) had passed the real guard and advanced to the next row.
All 406 old served evidence/audio/summary files still matched their archive
hashes after UI deployment. The existing three-hour monitor was extended to
verify and publish this pilot as well as finish the Simul baseline; no duplicate
automation was created and no schedule change was made.

## Failure audit, 2026-09-07

This pilot is NOT complete. An independent compute-node validation against the
frozen TSV and decoder settings found 18 valid utterance JSONs, 32 missing, and
zero invalid completed JSONs. Valid counts by logical worker0/1/2/3 are5/6/1/6.
No `pilot_summary.json` or `review/READY.json` exists. Do not publish partial
outputs as a completed 50-case comparison or compute a selected-subset headline.

Slurm accounting shows workers10345238_0/_1/_2 FAILED with exit1. Worker3 is
currently Priority pending with Restarts2; its log also contains a failed prior
attempt. The report10345239 still depends on successful completion of the array.
Original failed workers must not be described as completed merely because they
disappeared from squeue.

Failure sites are rows6,20,25,27. All four recorded exceptions come from the
Qwen3.8 sampler response being rejected by the strict v3 grouped parser:
`Invalid candidate line in future group: plausible`, or, for row25,
`Empty future group must explicitly say None: contrastive`. This is not evidence
of an OOM or that Gemma alone failed. The response text and finish reason are not
included in the raised error, so the precise malformed text/truncation cause has
not yet been established. Complete successful candidates in earlier steps are
not the raw failed response.

Preserve all18 validated outputs and frozen old/new runtime artifacts. Before
resuming, diagnose the malformed response with explicit error-response logging;
do not silently accept malformed groups as valid empty abstentions, weaken the
comparison, or blindly requeue deterministic failing work. Any repair must record
its runtime difference and preserve the total24-GPU limit, including the newly
queued low-only training10345494_2. Repair dependencies must include all missing
logical work before report/publication can pass.

## Parser failure and resubmission (2026-09-07)

All four decode tasks of `10345238` died with `Invalid future_set_v3_suffix_icl response
from qwen38-sampler` after 18 of 50 cases (rows 6, 20, 25 and 27 were the crash points;
row 27 failed identically on three requeues because the sampler seed is deterministic).
No raw response was preserved by the old code, so the exact prefixes were re-queried with
the same seed formula on a 1-GPU probe (`experimental/probe_v3_sampler_format.py`, jobs
10345588 and 10345618, 176 responses from both samplers). Cause: on very short prefixes
(`I`, `A`, `Most sure and`) Qwen3.8 writes the second heading as `Contrast` instead of
`Contrastive`; the strict parser then treated it as a prose line under Plausible. The
fourth crash was a response that ended right after the `Contrastive` heading
(`finish_reason=stop`), rejected as "empty group must say None". Gemma never failed.

Fix (runtime files replaced in place, hashes recorded under `resubmissions` in the run
manifest; old copies kept as `*.pre-<timestamp>`):
- `ambiguity_sampler_prompt.py`: `parse_grouped_future_output` tolerates benign format
  variations (decorated or shortened headings, preamble, `<think>` blocks, `1)`/`(1)`/bullets,
  bold or quoted items, `None.`/`(none)`/`N/A`/numbered `None`, an empty group followed by
  the other heading, and a trailing empty group only when `finish_reason == "stop"`).
  It still raises on missing/repeated headings, candidates before a heading, prose inside a
  group, mixing `None` with items, duplicate numbers, over-budget groups, and truncation.
- decoder: a malformed response is resampled up to `--targeted-parse-retries` (2) times with a
  shifted seed; every malformed response is written verbatim to
  `<verbose-dir>/malformed_sampler_responses.jsonl` (recovered or not) and to stderr; only if
  every attempt is malformed does the case fail.
- `run_single_case_ab.sbatch`: a failed case writes `FAILED.txt` and the task continues with
  its remaining rows, then exits non-zero so the report job does not run on a partial set.
All 176 probe responses parse under the new parser (mean 9.5 candidates per sampler,
one legitimate all-`None` abstention); 37 unit tests pass.

Resubmitted 2026-09-07 14:35 UTC: generation `10345647` (array 0-3, `--skip-existing`
keeps the 18 verified cases), report `10345648` (`afterok`). Superseded report `10345239`
cancelled. GPU budget unchanged: Simul generation 16 + pilot 8.

## Completed and published (2026-09-07 15:42 UTC)

Generation `10345647` finished all 32 missing cases (tasks 0/1/3 decoded 20–32 min each; task 2
found its rows already verified and skipped), report `10345648` validated 50/50 outputs and built
the bundle. The retry path fired twice and recovered both times, so no case was lost; the raw
responses are in `row_28/.../malformed_sampler_responses.jsonl` (Qwen3.8 heading garbled to
`Contrastone` on `No, I'm not coming any`) and `row_38/...` (on the ASR-broken prefix `I did nt`
Qwen3.8 started reasoning in prose and hit `max_tokens`; `finish_reason=length`, correctly rejected).
Archive: `data_synthesis/outputs/trajectory_reviews/archives/suffix-icl-v3-50cases-20260907-130701`
(gitignored, 160 files, sibling `.sha256.json`). Published with `manage_experiments.py publish`
into the served comparison root; `http://127.0.0.1:8768/experiments.html` now shows all three arms.

Same 50 cases, same models/seed/boundary flags; pilot metrics are synthesis-time char-BLEU vs the
frozen LLM reference and word-based text LAAL, not speech-model results:

| arm | mean char-BLEU | mean word LAAL | write steps |
|---|---|---|---|
| baseline (v2 prompt, dataset-unit window) | 45.02 | 7.33 | 10.98 |
| source-only boundary (v2 prompt) | 46.95 | 6.23 | 10.82 |
| **suffix-ICL v3** (v3 prompt, same boundary flags) | 44.88 | 5.62 | 11.54 |

Paired against source-only boundary: BLEU improved in 16, regressed in 30, tied in 4 (mean −2.07);
LAAL improved in 39, regressed in 10, tied in 1 (mean −0.61); both better in 14, both worse in 8.
Largest BLEU regressions: `AUD0000000140_489` 57.3→39.0, `AUD0000000003_1015` 66.0→51.7,
`AUD0000000086_115` 52.1→37.9. Against the historical baseline v3 is BLEU-neutral (−0.14, 22 up /
26 down) and clearly earlier (LAAL −1.70, 42 up / 7 down). Reading: the v3 prompt (literal-suffix
fit, fewer but stricter candidates, mean 9.5 per sampler instead of 20) makes the decoder commit
earlier, and on this set that costs synthesis-time BLEU relative to the v2 prompt with the same
boundary fixes. Per the rank-by-COMET rule, synthesis BLEU is not the decision metric; inspect the
regressions in the viewer before deciding whether v3 goes to a 40k run.

## Review fixes (2026-09-07, after the pilot)

Haoling's code review found two holes in the retry path, both reproduced and now fixed with
regression tests (42 tests pass):
- P1, truncated responses: any sampler response with `finish_reason == "length"` is rejected by
  `parse_grouped_future_output`, whichever groups it filled; a cut-off item such as
  `because she wanted to` no longer passes as a future.
- P2, lost evidence: the retry loop is now `sample_grouped_futures(request, n, retries, record)`
  in `ambiguity_sampler_prompt.py`; every malformed response is written to
  `malformed_sampler_responses.jsonl` the moment it is rejected (`event=malformed`), followed by
  one `recovered` or `exhausted` line, so a later request exception cannot lose it.
The decoder keeps the original exception on `--targeted-fail-on-api-error` and the
`Invalid <version> response from <model>` message. The pilot above ran with the earlier version
(runtime hashes in the manifest); the fixed code is what any future pilot or 40k run stages.
Decision unchanged: no v3 40k, no training-recipe change; next is a 200-case held-out comparison
of v2-boundary vs v3 with XCOMET and step-level early-commit checks, candidate count controlled.

## JSON-schema sampler output (2026-09-07 evening, commit dbfac56)

Haoling asked for the root-cause fix instead of a tolerant parser: the v3 request now sends
`structured_outputs={"json": schema}` to vLLM (0.19.1 in the sampler env), the prompt asks for
`{"plausible": [...], "contrastive": [...]}`, and parsing is `json.loads` plus key/type/budget
checks. The heading/bullet/`None` tolerance layer is deleted; the 3-attempt retry with immediate
raw-response logging and the `finish_reason=length` rejection stay as the safety net.

Checks (no change to the frozen 40k method):
- Probe `10349882` (same 84 prefixes as the text-format probes, both samplers): 80/80 replies
  parse, all `finish_reason=stop`, no HTTP errors. Candidates per reply: Qwen3.8 9.2 (text
  format 13.6), Gemma 5.3 (5.6). The schema does not pad, so Qwen's lists got shorter.
- End-to-end decode of pilot rows 0–7 (`10349883`, 2 GPUs, 24 min, output root
  `consensus_decoding_pilots/v3json-8cases-20260907T2035Z`): 8/8 cases complete, 0 failed,
  0 malformed replies. Same seed and flags as the text-format v3 arm.

| arm (same 8 cases) | mean char-BLEU | mean word LAAL | raw candidates / step |
|---|---|---|---|
| baseline | 45.6 | 6.31 | – |
| source-only boundary (v2 prompt) | 54.8 | 5.58 | 40.0 |
| v3 text format | 48.3 | 5.15 | 18.4 |
| **v3 JSON schema** | 50.8 | 4.99 | 13.5 |

Per case the JSON arm beats the text-format arm on 1015 (66.0 vs 51.7; commits `叉子飞`,
then closes the sentence with `成了十几块碎片。` at the period, no early `进了`), 140_489 and
071_590, and loses on 225_11 (35.1 vs 47.2). Both v3 arms stay below source-only boundary on
BLEU on these 8 cases while committing earlier. The JSON format is now the code path; the
open question from the review stands: candidates per step (40 → 18 → 13.5) changes how easily
strict consensus forms, so any v2-vs-v3 comparison must control the candidate budget.
The 8 JSON-schema cases are published as a fourth, partial arm `suffix_icl_v3json` in the served
comparison root (`manage_experiments.py add-arm`, commit 2225fc4; the UI now accepts partial arms
and labels the 42 cases that were not generated). Compare v3 text vs v3 JSON at
`http://127.0.0.1:8768/experiments.html#case=AUD0000000003_1015&left=suffix_icl_v3&right=suffix_icl_v3json`;
trajectories at `http://127.0.0.1:8768/suffix_icl_v3json/index.html#case=<utt_id>`.

## Combined changes test (2026-09-07 night, commit b1d54b1, job 10351383)

Implemented as opt-in decoder flags, all off by default:
- `--probe-input-mode heard-guessed`: probe prompt `[TASK] Translate the [HEARD] English … [HEARD] observed
  … [IMPORTANT] translate only what was heard; the continuation is one plausible guess … [POSSIBLE
  CONTINUATION] future`. Shared text first, future last, so prefix caching covers everything but the
  future and the committed Chinese.
- `--min-voters-abs N`: absolute voter floor on top of `--min-voters-ratio`; with N=10 a step with
  fewer than 10 surviving futures cannot commit.
- `--contrastive-notes`: v3 JSON schema makes each contrastive item `{"suffix", "resolves"}`; the note
  is stored in the audit (`note`) and stripped before probing.
- Non-method: Gemma and Qwen sampler calls run concurrently; suffix cap 200 → 120 characters;
  `[Timing] sampling= / probe_batches= total= / completion=` lines in every verbose log.
Test: pilot rows 0–9, same seed 1015, all three method flags on, output root
`consensus_decoding_pilots/v3json-all-10cases-20260907T2327Z`, 2 GPUs. Compare against the v3-JSON
arm (rows 0–7 done, rows 8–9 from array 10350658) and source-only boundary. The 42-case v3-JSON
array 10350658 is still running (tasks preempted and requeued twice).

## v3-JSON on all 50 cases (array 10350658 completed 2026-09-08 00:56 UTC)

50/50 complete, 0 failed, 0 malformed replies. Published as the complete `suffix_icl_v3json` arm
(replacing the 8-case partial one); archive `data_synthesis/outputs/trajectory_reviews/archives/v3json-50cases-20260907`.

| arm (50 cases) | mean char-BLEU | mean word LAAL |
|---|---|---|
| baseline | 45.02 | 7.33 |
| source-only boundary (v2 prompt) | 46.95 | 6.23 |
| v3 text format | 44.88 | 5.62 |
| **v3 JSON schema** | 45.73 | 5.38 |

Paired: vs source-only boundary BLEU −1.21 (19 up / 23 down), LAAL −0.86 (42 earlier / 7 later);
vs v3 text +0.85 BLEU (23 up / 24 down), LAAL −0.24; vs baseline +0.71 BLEU, LAAL −1.95.
Largest losses vs source-only: `AUD0000000003_1125` 63.7→54.5, `AUD0000000225_11` 44.4→35.1,
`AUD0000000140_489` 57.3→49.3. Reading: the JSON format recovers most of the text-format v3 loss
and is the earliest arm, but on synthesis BLEU it still sits 1.2 below the v2 source-only arm; the
combined-changes test (heard-guessed probe, voter floor 10, contrastive notes) is the next data point.
First submission of that test (10351383) failed in `pilot_case_guard.py`, which rejected the new
flags; allowlist extended and resubmitted as `10352105` (rows 0–9, root
`v3json-all-10cases-20260908T0059Z`).

## Combined-changes result (job 10352105, 2026-09-08 01:38 UTC): negative

10/10 complete, 0 failed, 0 malformed. With heard-guessed probe + voter floor 10 + contrastive
notes on together, the same 10 cases fall from 49.3 to **36.9** char-BLEU while LAAL drops from 5.40
to **3.14** (9 of 10 cases lose BLEU; 8 of 10 commit earlier; write steps 11–18 per case vs 6–12).
The predictions are literal, word-order-preserving translations committed chunk by chunk, e.g. 1015:
`还有叉子飞进了十几块碎片。这个巨人甚至更愤怒比第一个，和词是刚刚来到殴打，当第三个巨人再次介入。`
Mechanism (from the trajectories): telling the translator to "translate only what was heard" removes
the signal that the sentence is still open. Every future then yields the same eager translation of
the heard prefix, the vote is unanimous, and the decoder commits a calque at almost every chunk. The
joined input, where the future is glued on, was doing real work: it makes the probe treat the
sentence as unfinished and produce fluent Chinese, and it is the disagreement between futures that
gates commits. The voter floor cannot help when every future agrees (11.3 accepted per step ≥ 10).
Published as the partial arm `v3json_all_changes` on the comparison site for inspection.

Measured time split (first run with `[Timing]`): sampling 1657 s over 179 steps (9.3 s/step, both
samplers in parallel), probes 154 s over 615 batches (0.25 s/batch), completions 4 s. Sampler
generation is 91 % of decode time; probes 8 %; completions negligible. `max_tokens` for completions
is irrelevant to speed; the only levers are sampler concurrency (production already runs 8 cases
per worker) and sampler output length.

Attribution: one-flag ablations on the same 10 cases submitted 2026-09-08 (`v3abl_floor10`,
`v3abl_notes`, `v3abl_heard`; roots `consensus_decoding_pilots/v3json-ablate-<flag>-10cases-*`).
Expected: the heard-guessed flag alone reproduces the collapse; floor and notes alone stay near
v3-JSON.

## Correction (2026-09-08): scope reset to speed-only plus the resolves field

Haoling's instruction: no changes to the sampling or translation logic; only speed work
(parallelism, prompt-prefix order, token caps), plus trying the `resolves` field on contrastive
candidates. Accordingly (commit after b1d54b1): `--probe-input-mode` and `--min-voters-abs` are
removed from the code; the `v3json_all_changes` arm is retired from the comparison site (the negative
result above stays on record); the three one-flag ablations were cancelled before producing results.
Kept: concurrent Gemma/Qwen sampler calls, `[Timing]` lines, `--contrastive-notes` (opt-in), and a
new speed-only `--probe-prompt-order shared-first` that reorders the probe prompt sections
([TASK][IMPORTANT][INPUT] instead of [TASK][INPUT][IMPORTANT]) with byte-identical wording so
prefix caching covers everything up to the future. Suffix cap back to 200 characters.
Two 10-case jobs on rows 0–9, seed 1015: `v3json_speed` (shared-first only; must reproduce the
v3-JSON arm up to probe jitter, and shows the timing gain) and `v3json_notes` (shared-first +
contrastive notes).

## Speed-only checks and the reproducibility finding (2026-09-08 04:00 UTC)

Runs (all v3-JSON flags, seed 1015, 0 failed cases, 0 malformed replies across 70 cases):
`v3json_speed50` (10352713, 50 cases, shared-first probe order), `v3json_speed` (10352650, rows 0–9,
shared-first), `v3json_repeat` (10352950, rows 0–9, historical order = identical config to the
reference v3-JSON arm), `v3json_notes` (10352651, rows 0–9, shared-first + `--contrastive-notes`).

| comparison | identical predictions | mean BLEU | mean LAAL |
|---|---|---|---|
| A. speed-only 50 vs reference v3-JSON | 4 / 50 | 45.56 vs 45.73 | 5.32 vs 5.38 |
| B. same-config repeat vs reference, rows 0–9 | 2 / 10 | 45.09 vs 49.34 | 5.27 vs 5.40 |
| C. two shared-first runs, rows 0–9 | 7 / 10 | 46.06 vs 45.45 | 5.40 vs 5.48 |
| D. notes vs reference, rows 0–9 | 1 / 10 | 49.62 vs 49.34 | 5.20 vs 5.40 |

Findings:
- **Reproducibility (B) is the real problem.** The identical configuration decoded twice agrees on
  2 of 10 predictions; per-case char-BLEU moves by up to 21 points (1015: 66.0 → 44.6) and the
  10-case mean by 4.25. The sampler side is seeded and byte-stable (raw candidate counts per
  sampler/group identical to one decimal in every run); the divergence comes from probe logprob
  jitter flipping low-margin unanimous votes, after which the trajectory never re-converges.
  Consequence: 10-case comparisons are noise, and 50-case mean differences of about 1 BLEU (the
  v3-JSON vs source-only gap) are inside run-to-run variation. Any decision needs repeats or a
  much larger set.
- **Speed.** Wall time per case 176 s vs 185 s (A) and 160 s vs 171 s (B): the 5–6 % gain is
  present with historical order too, so it comes from the concurrent Gemma/Qwen calls. The
  shared-first reorder gives no measurable probe gain (0.184 s/batch vs 0.163 s/batch historical
  in B; probes are < 8 % of time). Sampling is 88 % of decode time in every run.
- **Shared-first may improve reproducibility** (C: 7/10 identical across two runs vs 2/10 for
  historical order in B), plausibly because the uncached tail per probe prompt is shorter. n = 10,
  one pair each; needs a second historical repeat pair before it counts.
- **Contrastive notes (D).** Quality unchanged within noise. Qwen's contrastive group shrinks
  from 3.8 to 2.0 items per step (Gemma 1.8 → 2.1), accepted futures 12.6 → 11.3; reasons are
  distinct in 243/248 groups, some genuine readings, some contrived; ~7 % slower per case.
Outputs: `consensus_decoding_pilots/v3json-{speed-50cases-20260908T0204Z,repeat-10cases-20260908T0256Z,speed-10cases-20260908T0158Z,notes-10cases-20260908T0158Z}`.
The notes run is published as the partial arm `v3json_notes` (10 cases). The viewer now carries the
`resolves` note per selected contrastive candidate (builder reads it from the raw block, page shows
it as a tag next to the candidate and in the translator-probe rows); 513 of 2,031 selected
candidates in the arm have one. Compare at
`http://127.0.0.1:8768/experiments.html#case=AUD0000000003_1125&left=suffix_icl_v3json&right=v3json_notes`.

## 35k production run submitted (2026-09-08, run tag `v3json-boundary-q38-gemma-q36-strict-35k-20260908`)

Config: v3 suffix-ICL prompt with vLLM JSON-schema output, source-only sampler context,
sentence-anchor window, sentence-end completion (conservative, match-source punctuation), no
`resolves` field, historical probe order; 35,000 rows, 24 decode tasks × 2 GPUs at 12 concurrent
(24-GPU cap), 16 cases per worker; SEGALE → MetricX QE ≤ 3.0 → length 0.7–1.5 → sample 12,500
(seed 42) → train → ACL + Simul-tst eval. Commit b357879 on `feature/home-checkout`, pushed and
pulled on BABEL. Smoke (job 10354000, 64 rows, 16 cases, one worker): 36 min, 0 failures, 0
malformed replies, ≈106 rows/hour/worker (40k run: ≈40 at 8 cases).
Pre-launch review (workflow, 3 lenses; 5 findings verified before a usage-limit stop) led to:
decode gate (exact 35,000 distinct loadable JSONs with the v3 prompt version) before SEGALE,
training-count gate (exactly 12,500 rows, real counts appended to the manifest) before training,
`--time-min 08:00:00`, current bad-node exclusions for decode and inference, pinned env values,
refusal on a dirty checkout or above the GPU cap, endpoint-attributed request failures, logged
sampler outages, atomic per-utterance JSON writes. Jobs: decode 10356205, decode_gate 10356206,
segale 10356207–10356209, qe 10356210–10356212, length 10356213, convert 10356214, train_gate
10356215, train 10356216, eval_launcher 10356217. Manifest:
`/home/haolingp/slurm_runs/v3json-boundary-q38-gemma-q36-strict-35k-20260908/run_manifest.txt`.
A 3-hourly babysit is scheduled in the Claude session (repairs limited to resubmitting dead decode
tasks and re-chaining; gates are never bypassed).

## Boundary-completion failures in the 35k run are translator repetition loops (2026-09-09)

Three rows of 35,000 died on the sentence-end guard
(`Boundary completion missing or token-truncated; refusing to fabricate a closed sentence`,
`force_complete_translation`): row 619 (task 2), row 639 (task 23) and row 942 (task 17).
Row 639 passed on retry; row 942 (`AUD0000000233_363`, global row 25,745) failed three times.

Reproduced on one GPU (job 10371576) at the exact failing step, chunk 4/28, observed source
`Will consist in keeping two or three of LUPIN's men busy.`, empty committed prefix, terminal `。`,
prompt 123 tokens. The translator returns a degenerate loop and never stops:

```
将 consists in keeping two or three of LUPIN's men busy. 将 consists in keeping two or three of
LUPIN's men busy. 将 consists in keeping two or three of LUPIN's men busy. ...
```

`finish_reason=length` at max_tokens 128, 256 and 512 alike, so **the token cap is not the cause and
raising `--final-max-tokens` does not help.** This is the same repetition-loop failure mode
documented on Simul-tst-COMMON, here inside the *synthesis* translator rather than the trained
model, triggered by a subject-less sentence fragment (`Will consist in ...`) that the ASR split
produced. The guard is behaving correctly: it refuses a degenerate completion instead of writing it.

Options (decision pending, no method change made): accept 34,999 of 35,000 rows and let the decode
gate allow that one named row; or add a repetition brake (presence/frequency penalty) to the
completion call only, which is a synthesis-method change and would need its own validation.
