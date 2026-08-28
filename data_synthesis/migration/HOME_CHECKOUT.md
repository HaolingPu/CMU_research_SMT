# Canonical code checkout: `/home/haolingp/CMU_research_SMT` (2026-08-28)

## Why
Your SLURM associations are now **preempt_qos / preempt_cpu_qos only** (no general/debug/cpu).
Login nodes cannot see `/data/user_data`, so a controller agent on your Mac that does
`ssh babel` needs the *code* somewhere the login node can read: `/home` (mounted on login
AND compute nodes). Data (outputs, ckpts, models, hf_cache) stays on `/data/user_data` and
resolves at job runtime on the compute node.

## Layout
| What | Where | Visible from |
|---|---|---|
| **Code (canonical)** | `/home/haolingp/CMU_research_SMT` — fresh clone of GitHub `feature/llm-wiki` + all untracked working files, branch `feature/home-checkout` | login + compute |
| Data / ckpts / models / caches / `.env` | `/data/user_data/haolingp/...` (unchanged) | compute only |
| Old working tree | `/data/user_data/haolingp` (git root, 966 GB junk `.git`) | compute only — now LEGACY, do not edit code there |

## What was changed in the code
A single mechanical rewrite in the `/home` clone (uncommitted, branch `feature/home-checkout`):
`/data/user_data/haolingp/{scripts,data_synthesis/codes,data_synthesis/codes-refactored}` →
`/home/haolingp/CMU_research_SMT/{...}` in every `*.sh`, `*.sbatch`, `*.py`
(723 references, 394 files). Kept on `/data` on purpose: `data_synthesis/codes/gigaspeech/hibiki/output/`
(3.7 GB data, 23 refs), `.env`, `datasets/`, `simul_tst_common/`, everything under `outputs/`,
`ckpts/`, `models/`, `hf_cache/`, `conda_envs/`, `code/`, `tools/`.
Log dirs referenced by `#SBATCH --output` were pre-created under `/home` (`scripts/{train,infer}/slurm_logs`,
`codes/gigaspeech/hibiki/hibiki-100`).
Verify: `grep -rE '/data/user_data/haolingp/(scripts|data_synthesis/codes)' --include=*.sh --include=*.sbatch --include=*.py . | grep -v hibiki/output` → empty.

## The agent loop (Mac = controller)
```bash
# Mac
git commit -am "..." && git push
ssh babel 'cd ~/CMU_research_SMT && git pull --ff-only'
ssh babel 'cd ~/CMU_research_SMT && sbatch scripts/train/train_X.sh'
ssh babel 'squeue -u haolingp'
ssh babel 'tail -50 ~/CMU_research_SMT/scripts/train/slurm_logs/<job>.out'   # logs live in /home now
ssh babel "~/bin/ondata 'ls /data/user_data/haolingp/ckpts/infinisst-omni/X'"   # only for /data files
```
`ondata` needs a running job to hop into; the old `anchor` job (cpu partition) is gone.
Re-create it on `preempt`: `sbatch --partition=preempt --qos=preempt_cpu_qos --requeue --time=2-00:00:00 --wrap 'sleep infinity' -J anchor`.

## Before the loop works — YOUR decisions
1. **Commit** the rewrite + the ~190 untracked files in `/home/haolingp/CMU_research_SMT`
   (52 MB, nothing >100 MB; `granary_en_1000h_manifest.tsv` 54 MB will trigger a GitHub size
   *warning* only) and push `feature/home-checkout`. Until then `git pull` on Babel would
   refuse (dirty tree). Nothing has been committed or pushed.
2. **Training must become preempt-resumable** (`--requeue` + resume from
   `latest_checkpointed_iteration.txt`) — see `PREEMPTION_READINESS.md` §A. Not done.
3. Optional: delete the 966 GB `.git` junk under `/data/user_data/haolingp` (not done).

## External code
Pinned commits + the local SEGALE patch/script live in `external/` (see `external/README.md`).

## Caveats
- `/home` quota: checkout is 247 MB; new `slurm_logs` and any results scripts write inside
  `codes/` now land on `/home` (96 GB used). Point heavy outputs at `/data` as before.
- Apptainer: `launch_container.sh` uses no `--contain`, so `$HOME` is auto-bound inside the
  container; `/data` is bound by site config (as before).
- `mac ← babel` sync now reads the `/home` clone directly (`PULL_TO_MAC.md`); the
  `migration_staging/` copy and `stage_to_home.sh` are superseded.
