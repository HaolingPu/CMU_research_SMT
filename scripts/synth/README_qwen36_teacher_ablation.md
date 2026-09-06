# Qwen3.6 teacher ablation

This harness answers one controlled question: did the translation teacher cause
the baseline improvement? It reruns EAST-even and Simul-MuST-C-fixed-v2 with
`Qwen3.6-35B-A3B-FP8` while keeping the student training recipe fixed.

## Fixed controls

- Baseline synthesis code is snapshotted from commit `9604507`, the code used
  for the February 2026 baseline artifacts. Only `--model-path` changes.
- Generation runs in `/data/user_data/haolingp/conda_envs/gemma4`, the same
  Qwen3.6-compatible vLLM runtime used by the completed ambiguity pipeline.
  The immutable historical snapshots are copied to a runtime directory where
  the renamed structured-output API and Qwen3.6's required 64-slot engine cap
  are applied. No prompt or decoding parameter is changed.
- Training uses exactly 12,500 rows, seed 42, and the unchanged
  `scripts/train/train_consensus_s.sh` Qwen3-Omni LoRA recipe.
- EAST keeps its historical MetricX threshold 3.0 and balanced split of
  4,166 low, 4,166 medium, and 4,168 high examples.
- Simul-MuST-C keeps its historical MetricX threshold 5.0.
- The manifest builder reuses historical audio chunks and multipliers. It
  replaces only assistant targets, maximizes overlap with the old selected
  12.5K examples, and records any necessary replacements.
- Both checkpoints run the same four segment sizes on ACL 6060 and
  Simul-tst-COMMON with the Standard inference prompt.

## Submit

Run from the BABEL checkout:

```bash
RUN_TAG=20260906-qwen36-teacher \
START_DEPENDENCY=<optional-afterok-dependency> \
PRIOR_EVAL_GATE=<optional-job-id> \
bash scripts/synth/submit_qwen36_teacher_baselines.sh
```

The submitter prints all job IDs and writes immutable code snapshots and run
manifests below `/home/haolingp/slurm_runs/qwen36-teacher-baselines-$RUN_TAG`.
Generated data stays in a distinct `qwen36_teacher_ablation` output tree, so
legacy baseline artifacts cannot be overwritten.

Before a production run, validate compute-node smoke outputs with
`python scripts/synth/validate_teacher_raw_output.py`; a successful generator
process exit alone is insufficient because the historical generators record
item errors in JSON.

Production DAGs include a raw-coverage gate after each generation array. The
gate requires one readable, unique-ID JSON file per input row, validates every
successful record's schema, and reports generator error records for downstream
filtering. It uses `afterany` so a time-limited array cannot silently block or
bypass coverage verification.
