---
title: Megatron-LM / SWIFT (training)
type: entity
tags: [tool, training, framework]
sources:
  - scripts/train/train_EAST_s.sh
  - scripts/train/launch_container.sh
created: 2026-06-01
updated: 2026-06-01
---

# Megatron-LM / SWIFT (training)

The training framework: SWIFT's Megatron CLI fine-tuning [[qwen3-omni]] with **LoRA** (rank 32,
alpha 32), MoE-aware configs. Standard hyperparameters across the ~23 variants: micro-batch 1 /
global-batch 4 on 4×L40S, LR 1e-4 (warmup 5%, min 1e-5), 1 epoch, save/eval every 200 steps,
audio chunks 15360 samples (960ms), max len 2048, full gradient checkpointing.

Runs in a Docker/modelscope image (`scripts/train/launch_container.sh`). Scripts:
`scripts/train/train_*.sh` (EAST, refined-EAST, Simul-MuST-C, LA, PA, consensus, hibiki, …).
Checkpoints export to `ckpts/infinisst-omni/<exp>/v<N>-…-hf/` (see [[infinisst-omni]]).

Note: zh runs use the **bundled** Megatron (no `MEGATRON_LM_PATH`); weak LA eval is data/policy,
not infra (train loss ~1.03 is non-predictive across runs).

## Related
- [[qwen3-omni]], [[infinisst-omni]], [[dataset-conversion-pipeline]], [[la-n-vs-wait-k]], [[babel-cluster]] (runs as 4×L40S `general` sbatch — mind the 2-day MaxTime).

## Sources
- code: `scripts/train/train_*.sh`, `launch_container.sh`
