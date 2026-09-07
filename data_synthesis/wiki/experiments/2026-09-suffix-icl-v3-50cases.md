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
