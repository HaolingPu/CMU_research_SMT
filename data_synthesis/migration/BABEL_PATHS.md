# Babel-only paths (do NOT copy to Mac)

All paths are on Babel; `/data/user_data` is mounted only on compute nodes with an active
job (AutoFS — `stat` the full path to trigger the mount). From a login node use
`~/bin/ondata '<command>'`.

## Code root (2026-08-28)

CODE_ROOT=/home/haolingp/CMU_research_SMT   # canonical checkout, visible on login+compute; all script/code paths rewritten to this
LEGACY_ROOT=/data/user_data/haolingp        # old working tree — data still lives here; code there is stale

## Checkpoints

```
CHECKPOINT_ROOT=/data/user_data/haolingp/ckpts                    # 3.1 TB
  infinisst-omni/<EXP_NAME>/                                      # per-experiment Megatron ckpts
  infinisst-omni/<EXP_NAME>/v<N>-YYYYMMDD-…-hf/                   # HF exports used for inference
  infinisst-omni/<exp>/v*-hf/evaluation/acl_6060/<lang>/seg<N>/   # eval results (scores.tsv)
  pretrained/                                                     # base ckpts
MENTOR_CKPTS=/data/user_data/siqiouya/ckpts/{infinisst-omni,pretrained}   # referenced by some scripts
```

## Model weights

```
MODEL_ROOT=/data/user_data/haolingp/models                        # 514 GB
  Qwen3-Omni-30B-A3B-Instruct/          # base model for all fine-tunes
  Qwen3-30B-A3B-Instruct-2507-FP8/      # main synthesis translator (most-referenced)
  Qwen3-30B-A3B-{Base,FP8,Thinking-2507-FP8}/
  Qwen3.5-122B-A10B-FP8/                # thinking future sampler
  Qwen3-4B-Base/  Qwen3.6-35B-A3B-FP8/  DeepSeek-R1-Distill-Qwen-32B/
  metricx-24-hybrid-xl-v2p6/  metricx-23-xl-v2p0/  mt5-xl/        # QE stack
  LaBSE/  gemma-4-E2B-it/  models--facebook--nllb-200-distilled-600M/
HF_CACHE=/data/user_data/haolingp/hf_cache                        # 127 GB
```

## Datasets / synthesis outputs

```
GIGASPEECH_SRC=/data/group_data/li_lab/siqiouya/datasets/gigaspeech/     # source manifests + audio (group share)
SWIFT_MANIFESTS=/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/
GROUP_SHARE_MINE=/data/group_data/li_lab/haolingp/data_synthesis/
SYNTH_OUTPUT_ROOT=/data/user_data/haolingp/data_synthesis/outputs/       # 1.8 TB — EAST / Refined_EAST / SALAMI / gigaspeech / mfa_*
ACL6060=/data/user_data/haolingp/datasets/acl_6060/                      # eval set (small; top-level datasets/ IS copied)
MFA_BACKUP=/data/user_data/haolingp/data_synthesis/MFA_backup/           # 54 GB
```

## Tooling / external code (re-clone on demand rather than mirror)

```
/data/user_data/haolingp/code/Megatron-LM      # external clone
/data/user_data/haolingp/code/OmniSTEval       # eval scorer used by eval_all_ckpts
/data/user_data/haolingp/tools/{SEGALE,metricx}
/data/user_data/haolingp/conda_envs/{evaluation,segale,gemma4,nb}
/home/haolingp/miniconda3/envs/{SMT,metricx,vllm,…}
/home/siqiouya/miniconda3 (envs omni_inference, consensus — mentor's, referenced by a few scripts)
```

## Secrets (NEVER copy)

```
/home/haolingp/.keys/wandb          # read by train_*.sh
/home/haolingp/.keys/huggingface
/data/user_data/haolingp/.env       # OPENAI_API_KEY, OPENAI_API_BASE, GPT_SAMPLER_MODEL,
                                    # GPT_REASONING_EFFORT, DEEPSEEK_API_KEY (names only; use .env.example)
```

## Where paths are hardcoded

Absolute `/data/user_data/haolingp/...` and `/data/group_data/...` paths are hardcoded
throughout `scripts/**/*.{sh,sbatch}`, `data_synthesis/codes/**`, and
`data_synthesis/codes-refactored/**` (~2000 occurrences; top hits: `data_synthesis/outputs`
859×, `codes` 444×, group gigaspeech datasets 215×, Qwen3-30B-FP8 119×, hf_cache 111×).
Consequence: **the code only runs on Babel as-is** — the Mac copy is for editing/reasoning,
and jobs must be synced back to the same Babel paths before submission (see
`migration/PULL_TO_MAC.md` and the tools/babel design in `PREEMPTION_READINESS.md`).
