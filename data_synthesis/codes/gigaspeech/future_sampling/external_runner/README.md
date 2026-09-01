# External second-half decode

This runner moves the second half of the 40K future-consensus **decode stage**
to another Slurm cluster without changing the method, prompt, filtering, or
consensus implementation.

## Work split

- BABEL: rows `0-20,003`.
- External cluster: rows `20,004-39,999` (`19,996` rows).
- The `20,004` boundary matches the original 12-task BABEL array exactly:
  BABEL task 5 ends at row `20,003`, and task 6 begins at `20,004`.
- External task directories start at `task_100`, so they can be copied into the
  main output root without colliding with BABEL's task directories.

Do not start both copies of the second half. Once the mentor confirms the
external job is ready, cancel BABEL decode array tasks `6-11` while leaving
tasks `0-5` running. The four rows `20,000-20,003` intentionally remain on
BABEL. Record the canceled component IDs in the BABEL run manifest.

Canceling those array elements means the old decode `afterok` dependency cannot
release the existing downstream chain. After the external outputs are returned
and all 40,000 rows are verified, cancel that stale downstream chain and
resubmit from SEGALE prepare. Do not resubmit or regenerate completed decode
outputs.

## Required environment

Use the exact Git commit recorded in `external_run_manifest.txt`, the exact
frozen TSV, and these checkpoints:

- Qwen3.8-27B-FP8: future sampler on GPU 0.
- Gemma-4-E2B-it: future sampler colocated on GPU 0.
- Qwen3.6-35B-A3B-FP8: translation/probe model on GPU 1.

Each array task needs two GPUs. GPU 0 must have about 46 GiB usable memory for
the colocated samplers; GPU 1 needs about 40 GiB. The tested software is Python
3.12 with the project `gemma4` environment and vLLM
`0.19.1rc1.dev28+g8617f8676`. Different GPU architectures or vLLM versions
should first run one small smoke slice.

The input TSV and model weights are not stored in GitHub. Transfer them to a
shared filesystem visible from the mentor's compute nodes.

## Submit

```bash
git clone git@github.com:HaolingPu/CMU_research_SMT.git
cd CMU_research_SMT
git checkout feature/home-checkout

export INPUT_TSV=/shared/input/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv
export OUTPUT_ROOT=/shared/output/ambiguity-half2
export PYTHON_BIN=/shared/envs/gemma4/bin/python
export QWEN38_MODEL=/shared/models/Qwen3.8-27B-FP8
export GEMMA_MODEL=/shared/models/gemma-4-E2B-it
export QWEN36_MODEL=/shared/models/Qwen3.6-35B-A3B-FP8

# Eight workers use 16 GPUs. Change these to match the mentor's allocation.
export NUM_TASKS=8
export MAX_CONCURRENT_TASKS=8
export PARTITION=gpu
# export ACCOUNT=my_account
# export QOS=my_qos
# export GPU_GRES=gpu:a100:2

bash data_synthesis/codes/gigaspeech/future_sampling/external_runner/submit_external_second_half.sh
```

For a smoke test, set `ROW_OFFSET=20004`, `SLICE_ROWS=10`, `NUM_TASKS=1`, and
`MAX_CONCURRENT_TASKS=1`. Remove those overrides before the full submission.
The worker is restart-safe: it skips existing per-utterance JSON files and a
task with `DONE.txt` exits immediately.

## Verify and return

After every task completes:

```bash
$PYTHON_BIN data_synthesis/codes/gigaspeech/future_sampling/external_runner/verify_external_slice.py \
  --input-tsv "$INPUT_TSV" \
  --output-root "$OUTPUT_ROOT" \
  --row-offset 20004 \
  --num-rows 19996
```

The last line must be `VERIFIED`. Return only the decode artifacts, not model
weights or caches:

```bash
rsync -av --partial "$OUTPUT_ROOT"/task_1*/ babel:/path/to/main-output-root/
```

Run the verifier again on the combined root before SEGALE, MetricX, conversion,
training, or evaluation, changing the final arguments to `--row-offset 0
--num-rows 40000`. Do not start downstream stages until the combined decode has
exactly 40,000 unique utterances.
