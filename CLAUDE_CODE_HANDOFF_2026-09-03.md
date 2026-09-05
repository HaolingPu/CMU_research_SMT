# Claude Code Handoff: Ambiguity-Aware Future-Consensus SMT

Use this document as the operational context for continuing Haoling Pu's
simultaneous English-to-Chinese translation experiment. Verify live cluster
state before acting because the status snapshot below is time-sensitive.

The maintained action list is `RESEARCH_TODO.md`; update it as experiments and
mentor-facing deliverables are completed.

## 1. Objective

The research goal is to reduce irreversible early commitment in simultaneous
machine translation. At each partial English ASR prefix, two future samplers
generate plausible and ambiguity-resolving continuations. A separate Chinese
translation/probe model predicts the next target token under every retained
future. The decoder writes only target tokens that survive strict consensus;
otherwise it reads more source context.

This is no longer framed as an arbitrary five-axis method. The current prompt
is generic and examples/constraints target real lexical, syntactic,
referential, discourse, and attachment ambiguity. The paper claim should be
about future-consistent decoding, not "Axis 5."

## 2. Repositories and Git State

### Research repository

- Local: `/Users/haolingpu/Desktop/research/CMU_research_SMT`
- BABEL: `/home/haolingp/CMU_research_SMT`
- Branch: `feature/home-checkout`
- Local/remote HEAD at handoff: `9832ad9`
- Worktree is clean except an untracked `.DS_Store`.
- Never discard generated outputs because many live artifacts are outside Git.

Important recent commits:

- `9832ad9`: document immutable generation image.
- `6ae6129`: fix pinned generation image dependency install.
- `37d23e9`: package generation runtime for external clusters.
- `a7e8f6b`: add 100-case trajectory quality audit tooling.
- `2b62a8a`: improve trajectory viewer navigation/search.
- `921c104`: add mentor-facing trajectory viewer.
- `b9e8fe3`: add portable external decode runner.
- `ab7405c`: revert unsafe parallel sampler startup.

### Official Open-LiveTranslate repository

- Local: `/Users/haolingpu/Desktop/research/Open-LiveTranslate`
- Remote: `git@github.com:LeiLiLab/Open-LiveTranslate.git`
- Contribution branch: `codex/future-consensus-external-runner`
- Branch/remote HEAD: `e828372764c736aa43eb0159798167af063c6d94`
- Pull request: `https://github.com/LeiLiLab/Open-LiveTranslate/pull/34`
- Base: `main`
- Verified at handoff: `e828372` is **not** an ancestor of `origin/main`, so the
  contribution is not merged. The remote branch exists. Check the PR UI for
  review/check status because the repository is private and local Git cannot
  report the web review state.
- Local `main` is stale; do not use it as the authoritative base. Fetch first.

The official contribution follows repository conventions:

```text
data/recipes/future_consensus_recipe.sh              # user-facing submitter
data/recipes/future_consensus_decode.sbatch          # two-GPU Slurm worker
data/scripts/s2t/future_consensus/consensus_decoding.py
data/scripts/s2t/future_consensus/ambiguity_sampler_prompt.py
data/scripts/s2t/future_consensus/verify_external_slice.py
data/scripts/s2t/future_consensus/README.md
data/envs/future-consensus/future_consensus.def
data/envs/future-consensus/requirements-freeze.txt
data/envs/future-consensus/README.md
tests/data/test_future_consensus.py
```

The recipe is intentionally standalone and does not alter the default
`data/recipes/gigaspeech_recipe.sh` pipeline. It accepts any contiguous range
through `ROW_OFFSET` and `SLICE_ROWS`; it is not hard-coded as "second half."

Verification completed at handoff:

- `python3 tests/data/test_future_consensus.py`: 11 tests passed.
- `bash -n` passed for the recipe and sbatch worker.
- `pytest` is not installed in the laptop's default Python, so use the official
  repo's `OLT_VENV_ROOT/olt-main/bin/python tests/data/run_tests.py
  future_consensus` for the repository-standard test command.

## 3. Frozen Method and Model Settings

Current production settings come from the BABEL run manifest:

```text
run_tag=ambiguity-q38-gemma-q36-fsetv2-prefixnorm-strict-40k-r1-20260831
prompt_version=future_set_v2_two_groups
prefix_normalization=case-insensitive-word-boundary
sampler_1=gemma-4-E2B-it
sampler_2=Qwen3.8-27B-FP8
translator_probe=Qwen3.6-35B-A3B-FP8
total_rows=40000
targeted_num_futures=20
plausible_per_sampler=10
contrastive_per_sampler=10
max_raw_candidates_per_prefix=40
min_voters_ratio=1.0
future_source_window=1
qe_threshold=3.0
length_ratio_ref=0.7:1.5
train_sample_n=40000
sample_seed=42
quality_metrics=BLEU,Unbabel/XCOMET-XL
```

`TARGETED_NUM_FUTURES=20` means **20 candidates per model**: 10 plausible and
10 contrastive. Across Gemma and Qwen there can be at most 40 raw candidates
per sampled source prefix. The samplers produce a coordinated numbered list in
one response so each model can avoid within-set duplication.

The canonical prompt is:

```text
data_synthesis/codes/gigaspeech/future_sampling/ambiguity_sampler_prompt.py
```

The official PR copy is:

```text
data/scripts/s2t/future_consensus/ambiguity_sampler_prompt.py
```

The filter rejects malformed output, prompt/meta leakage, overlong text,
repeated observed prefixes, exact duplicates, and near duplicates. Prefix
normalization is a post-generation repair/filter: it strips a repeated observed
prefix at a case-insensitive word boundary when the remaining suffix is a valid
continuation. It does not alter the source or force candidates to agree.

Strict consensus uses `MIN_VOTERS_RATIO=1.0`. The probe translates under every
retained future and commits only a next token supported by all retained futures.
This intentionally trades latency for early-commit safety.

## 4. Runtime and GPU Layout

### Standard two-GPU worker

- GPU 0: Qwen3.8-27B-FP8 at memory utilization `0.70` plus
  Gemma-4-E2B-it at `0.23` with `--enforce-eager`.
- GPU 1: Qwen3.6-35B-A3B-FP8 at `0.85`.
- All model servers use maximum model length 4096.
- Qwen/Gemma `max-num-seqs=16`; translator `max-num-seqs=64`.
- Decode concurrency was increased from 4 to 8, then selected workers used 12.
- Output is restart-safe through per-utterance JSON existence checks and task
  `DONE.txt` sentinels.

### Optimized three-GPU canary

- Two logical sampler workers use GPUs 0 and 1.
- They share one Qwen3.6 translator on GPU 2.
- Two logical ranges are decoded concurrently with 12 cases per worker.
- Stable measured throughput: 107.5 rows/hour steady state versus 62-75 for two
  old workers, a 1.43-1.73x steady-state speedup.
- Sampler generation: 90.4-92.9 tokens/s; translator: 68.9 tokens/s.
- Translator queue peaked at 276 waiting requests, so it is the bottleneck.
- No HTTP, OOM, or engine failures in the measured stable window.
- An attempt to start Qwen and Gemma fully in parallel produced zero Qwen KV
  cache and was reverted in `ab7405c`. Do not restore it blindly.
- Prefix caching was also tested and reverted because Qwen KV cache fell from
  2.96 GiB to 0.8 GiB and throughput degraded.

Never exceed 24 concurrent GPUs. Current intended peak is 23 GPUs.

Confirmed GPU-node exclusions for new submissions:

```text
babel-o9-24,babel-q9-32,babel-t5-28
```

Before resubmitting GPU work, inspect `scontrol show node`, recent scheduler
state/reason, and `#babel-babble` if Slack access exists. Do not add stale
exclusions when a node is live-healthy and has successfully run this pipeline.

On BABEL, new GPU repairs must use `preempt/preempt_qos`; CPU-only repairs must
use `preempt/preempt_cpu_qos`. Preserve all completed JSONs and sentinels.

## 5. Reproducible Environment

The generation environment is separate from alignment, MetricX, training, and
evaluation environments. Git tracks definitions, not large binaries or data.

Exact generation runtime:

```text
Python 3.12.13
vLLM 0.19.1rc1.dev28+g8617f8676
PyTorch 2.10.0+cu129
Transformers 5.5.0
pandas 3.0.2
NumPy 2.2.6
tokenizers 0.22.2
safetensors 0.7.0
```

Portable image published through GHCR:

```text
docker://ghcr.io/haolingpu/cmu-research-smt-generation@sha256:f52a26baf96d561e8d80cf645f4d07a237f9d7c2a7f335fb19e341d13472f984
```

Research-repo documentation:

```text
containers/generation/README.md
data_synthesis/codes/gigaspeech/future_sampling/external_runner/README.md
data_synthesis/codes/gigaspeech/future_sampling/external_runner/GENERATION_IMAGE.txt
```

Official-repo environment definition:

```text
data/envs/future-consensus/future_consensus.def
data/envs/future-consensus/requirements-freeze.txt
```

The image contains software only. The following are **not** in Git or in the
image and must be supplied/mounted by the target cluster:

- Qwen3.8, Gemma-4, and Qwen3.6 model weights.
- Frozen input TSV.
- Hugging Face cache.
- Output directories.
- Credentials.
- Built Apptainer `.sif` (unless pulled from GHCR and cached locally).

The official submitter fingerprints the input TSV, each model `config.json`,
the environment definition/freeze, and SIF metadata. Reference SHA-256 values
are documented in `data/scripts/s2t/future_consensus/README.md`; do not set
`ALLOW_REFERENCE_MISMATCH=1` unless intentionally running a different
experiment.

## 6. Input and Output Contract

Frozen input TSV on BABEL:

```text
/data/group_data/li_lab/haolingp/consensus_handoff/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv
```

Columns:

```text
id  audio  n_frames  speaker  src_text_full  src_lang  tgt_lang
src_trajectory  asr  src_text  llm_reference_text
reference_source  reference_chars
```

Important semantics:

- `src_trajectory` is the incremental ASR/source sequence consumed by decode.
- `llm_reference_text` is evaluation/reference context, not the future sampler's
  privileged input.
- One successful row produces one per-utterance JSON plus a verbose log.
- JSON filenames are the global utterance IDs and are used for skip/verification.

Production decode root:

```text
/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/ambiguity-q38-gemma-q36-fsetv2-prefixnorm-strict-40k-r1-20260831
```

Post-processing root:

```text
/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/ambiguity-q38-gemma-q36-fsetv2-prefixnorm-strict-40k-r1-20260831-segale
```

Run manifest, always read first:

```text
/home/haolingp/slurm_runs/ambiguity-q38-gemma-q36-fsetv2-prefixnorm-strict-40k-r1-20260831/run_manifest.txt
```

`/data/user_data` and `/data/group_data` may not be mounted on the login node.
Count or inspect decode outputs from a compute node.

## 7. Active BABEL State at 2026-09-03 21:35 ET

Snapshot only; re-query before acting.

- Unique per-utterance JSONs: **27,667 / 40,000 (69.2%)**.
- Recent aggregate throughput: approximately 440-450 unique rows/hour.
- Rough decode ETA at that rate: approximately 27-28 hours.
- No current OOM, vLLM engine, HTTP, or ECC error was found in active logs.
- Main decode array: `10280652`.
- Dual-sampler replacement: `10298363` on logical ranges 0-2499 and
  2500-4999, writing `task_12` and `task_13`.
- `10282275` timed out normally and is obsolete; never resubmit it.
- Original chain `10274592-10274602` was intentionally canceled for tuning;
  never resubmit it.
- Main array tasks 0 and 1 are intentionally held during the dual worker.
- Other main-array tasks are running.
- Current GPU peak is 23.

Active downstream IDs, all currently waiting on decode:

```text
10280664  SEGALE prepare
10280665  SEGALE align, 24 GPU shards
10280666  SEGALE merge
10280667  MetricX input/QE prepare
10280668  MetricX QE, 24 GPU shards
10280669  QE finalize
10280670  length-ratio filter
10280671  convert to ms-swift training data
10280672  LoRA training
10280673  inference/evaluation launcher
```

### Critical dependency caveat

The downstream chain still depends on `afterok:10280652_*`, while main-array
tasks 0 and 1 are held... The dual worker does not satisfy that Slurm array
dependency. Do not assume the pending chain will release automatically.

Also note the exact row geometry:

- Main array uses `ceil(40000 / 12) = 3334` rows per task.
- Main task 0 covers 0-3333.
- Main task 1 covers 3334-6667.
- Dual worker covers only 0-4999.

Therefore rows 5000-6667 must still be completed without regenerating the
dual-worker overlap. Before post-processing, explicitly verify the full
0-39,999 range, repair only missing rows, then replace/repair downstream
dependencies. Do not release task 1 blindly if that would regenerate rows
3334-4999, and do not start SEGALE merely because a raw `find | wc -l` reaches
40,000; use the verifier to detect missing, duplicate, unexpected, or unreadable
utterances.

## 8. Full Pipeline After Decode

The complete experiment is:

1. Future-consensus decode to 40,000 per-utterance JSONs.
2. Verify exactly the intended 40,000 unique utterance IDs.
3. Convert outputs to 24 SEGALE shards.
4. Run SEGALE alignment on 24 single-GPU tasks. Partial aligned files resume by
   document ID; `ALIGN_DONE.txt` is the clean-completion sentinel.
5. Merge the 24 alignment outputs.
6. Build 24 MetricX QE shards.
7. Run MetricX-24-Hybrid-XL-v2p6 in QE mode, 24 single-GPU tasks.
8. Merge scores and retain an utterance only if the maximum per-sentence QE
   score is at most 3.0.
9. Apply reference-length ratio filter `0.7 <= ratio <= 1.5`.
10. Convert surviving JSONs and audio chunks to ms-swift JSONL, sampling up to
    40,000 rows with seed 42.
11. LoRA-fine-tune Qwen3-Omni-30B-A3B-Instruct with Megatron-SWIFT.
12. Export the newest Megatron adapter checkpoint to Hugging Face format.
13. Run SimulEval inference/evaluation, reporting BLEU, latency/LAAL, and
    Unbabel/XCOMET-XL.

Relevant files:

```text
data_synthesis/codes/gigaspeech/future_sampling/scripts/segale/run_prepare_shards_24.sbatch
data_synthesis/codes/gigaspeech/future_sampling/scripts/segale/run_segale_align_24gpu_preempt.sbatch
data_synthesis/codes/gigaspeech/future_sampling/scripts/segale/run_qe_prepare_24.sbatch
data_synthesis/codes/gigaspeech/future_sampling/scripts/segale/run_metricx_qe_24gpu_preempt.sbatch
data_synthesis/codes/gigaspeech/future_sampling/scripts/segale/run_qe_finalize_24.sbatch
data_synthesis/codes/gigaspeech/future_sampling/scripts/segale/run_length_ratio_filter.sbatch
scripts/train/run_convert2swift_consensus.sbatch
scripts/train/train_consensus_s.sh
scripts/infer/run_infer_after_train_generic.sbatch
```

MetricX model paths on BABEL:

```text
/data/user_data/haolingp/models/mt5-xl
/data/user_data/haolingp/models/metricx-24-hybrid-xl-v2p6
```

Training configuration:

```text
base=/data/user_data/haolingp/ckpts/pretrained/llm/Qwen3-Omni-30B-A3B-Instruct-mcore/
train_type=lora
lora_rank=32
lora_alpha=32
target_modules=all-linear
freeze_llm=false
freeze_vit=true
freeze_aligner=true
GPUs=4 L40S
expert_model_parallel_size=4
micro_batch_size=1
global_batch_size=4
lr=1e-4
warmup_fraction=0.05
min_lr=1e-5
weight_decay=0.01
max_epochs=1
max_length=2048
attention_backend=flash
```

Do not report final training/evaluation results until all downstream jobs and
artifacts are complete. Record survivor counts after both filters.

## 9. Pilot and Quality Findings

### Pilot progression

- Initial 50-case no-thinking pilot: mean char-BLEU 48.601, mean LAAL 5.954,
  no meta leakage, but all 50 cases had at least one too-few-future step.
- Coordinated 10+10-per-model pilot: mean char-BLEU 51.261, LAAL 5.513, no
  meta leakage, 7/10 cases with a too-few step. Candidate uniqueness was near
  1.0 and logs identify model plus plausible/contrastive mode.
- Prefix-normalized pilot: mean char-BLEU 49.609, LAAL 5.683, no meta leakage,
  8/10 cases with a too-few step. Prefix normalization recovered valid repeated
  continuations but did not itself improve aggregate BLEU.

### 100-case trajectory audit

Tool:

```text
data_synthesis/tools/trajectory_viewer/analyze_review_quality.py
```

Audit results:

- 100 cases; 2,560 trajectory decisions.
- 1,328 READ and 1,232 WRITE decisions.
- Confirmed early-commit failures: 3/100, cases 9, 71, 83.
- Boundary-ambiguity failure: 1/100, case 64.
- Separate MT semantic errors: 6/100.
- Watch/data/latency cases: 7/100.
- Mean/median char-BLEU: 49.30/48.75.
- Mean/median LAAL: 6.45/6.28.
- Mean selected futures per sampled step: 32.0; median 36.
- Gemma keep rates: 70.0% plausible, 72.0% contrastive.
- Qwen3.8 keep rates: 90.4% plausible, 88.1% contrastive.

Key failures:

- Case 9, `AUD0000000003_1011`: committed inanimate Chinese pronoun `它们` for
  a referent later resolved as people/giants. Sampler bias, not candidate count,
  because 36 futures survived.
- Case 71, `AUD0000000003_1125`: prefix reset lost antecedent/gender context and
  committed male `他` before later female resolution.
- Case 83, `AUD0000000003_1152`: prefix reset after `must remain no` lost
  negation and committed the opposite meaning.
- Case 64, `AUD0000000003_111`: punctuation-poor ASR hid a speaker boundary;
  futures failed to include speaker-switch/reported-speech alternatives.

Conclusion: the method is useful and often delays content-bearing decisions,
but early commitment is not solved. More futures alone will not address the
observed failures. The next research fixes should carry unresolved context
across normalization boundaries, add gender/antecedent uncertainty constraints,
and sample dialogue-boundary alternatives for punctuation-poor ASR. Evaluate
early commitment directly with human/oracle annotation; BLEU/COMET alone cannot
identify irreversible timing errors.

The 100 cases are contiguous from one recording and are not representative.
Build a stratified audit set before making a paper-wide percentage claim.

## 10. Trajectory Viewer

Source:

```text
data_synthesis/tools/trajectory_viewer/
```

It provides a searchable 100-case sidebar, compact source/translation chunk
view, READ/WRITE actions, selected futures by model/mode, final prediction,
reference, metrics, and optional audio.

Full shared bundle intended on BABEL compute nodes:

```text
/data/group_data/li_lab/haolingp/data_synthesis/trajectory_reviews/ambiguity-q38-gemma-q36-first100
```

`/data/group_data` is not mounted on `login2`; inspect it from a compute node.

Local stripped static bundle:

```text
/private/tmp/trajectory-viewer-v2-20260902
```

It currently contains only `index.html`, `styles.css`, `app.js`, and
`data/review.json` (about 6.9 MiB total). It was served locally at:

```text
http://127.0.0.1:8766/
```

The local URL is not shareable with the mentor. Public/cloud deployment was
requested but had not been completed at handoff. Before publishing, package
only the viewer-required assets, decide whether research examples may be public,
and avoid uploading raw verbose logs or credentials. If audio is required,
deploy the full approved bundle or use an authenticated/internal host.

## 11. Immediate Next Actions for Claude

1. Read this handoff and the live BABEL `run_manifest.txt` before any cluster
   action.
2. Re-query `squeue`, `sacct`, `scontrol`, active logs, and unique output IDs
   from a compute node. Do not treat Priority/Resources/Dependency/healthy
   preemption as failure.
3. Keep total GPU use at or below 24 and preserve all outputs/sentinels.
4. Monitor `10280652` and `10298363`; never resubmit obsolete jobs
   `10274592-10274602` or `10282275`.
5. When the current workers finish, compute exact missing row IDs. Specifically
   resolve rows 5000-6667 without regenerating the overlapping 3334-4999 range.
6. Verify exactly rows 0-39999 with the verifier, not only a file count.
7. Replace or repair the stale downstream dependency chain because the dual
   worker cannot satisfy `afterok:10280652_*` while array tasks 0/1 remain held.
8. Run SEGALE, MetricX, filtering, conversion, training, and evaluation in
   order; verify 24/24 shard outputs and survivor counts at each stage.
9. Report final BLEU, LAAL/latency, XCOMET, survivor counts, and training
   checkpoint provenance.
10. If asked to share the viewer, deploy the sanitized static bundle and return
    a real HTTPS URL; do not claim `127.0.0.1` is shareable.

## 12. Non-Negotiable Safety Rules

- Never rerun completed decode cases, SEGALE shards, or MetricX shards
  unnecessarily.
- Never exceed 24 concurrent GPUs.
- Never run post-processing before exact 40,000-row verification.
- Never use a moving image tag for the reproducible experiment; use the pinned
  digest.
- Never put tokens, model weights, datasets, SIF images, caches, or generated
  outputs in Git.
- Never remove a checkout used by queued/running Slurm jobs.
- Pull `feature/home-checkout` with `--ff-only` before using changed scripts on
  BABEL.
- Append every replacement job ID, failed node, reason, and preserved-output
  decision to the run manifest.
- Distinguish instruction-following/candidate quality, translation quality,
  and early-commit timing quality in all analysis.
