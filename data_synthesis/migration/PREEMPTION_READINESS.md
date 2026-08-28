# Preemption readiness + future Mac→Babel control design

## A. Preemption / requeue audit (no changes made)

### Training (`scripts/train/train_*.sh`) — NOT preemption-safe

Evidence (e.g. `train_EAST-latency2mult_s.sh`, `train_Simul-MuST-C_s.sh`):

- `#SBATCH --partition=general`, `--time=1-00:00:00`; **no `--requeue`**.
- `--finetune true` with `--load …/Qwen3-Omni-…-mcore/` (the pretrained base) and
  `--save ckpts/infinisst-omni/${EXP_NAME}` , `--save_interval 200`.

Consequences:

1. **Restart = step 0.** `--finetune true` loads only base weights, not optimizer/iteration
   state; there is no `--load` pointing at the experiment's own save dir. Resubmitting the
   same script re-trains from scratch.
2. **Same `--save` dir on rerun** → a second run writes new `iter_*` dirs alongside the old
   ones; a later HF export can pick up a mixed/ambiguous "latest" checkpoint. Reruns of an
   `EXP_NAME` should use a fresh name (current practice, e.g. `-bsz4` suffixes, effectively
   does this manually).
3. general-partition `normal` QoS is preemptable by `preempt_qos`/`array_qos` per the QoS
   table (wiki `babel-cluster`), and the 2-day MaxTime is a hard wall — a killed run loses
   everything past nothing (checkpoints every 200 steps survive) but cannot resume.

Mitigation to implement later (not now): add a resume mode — if
`ckpts/infinisst-omni/$EXP_NAME/latest_checkpointed_iteration.txt` exists, switch
`--load` to the save dir and drop `--finetune`; add `#SBATCH --requeue`. Until then,
**keep training on `general` and treat 1 epoch/2 days as the budget** (current runs fit).

### Synthesis / QE (`data_synthesis/codes/gigaspeech/east/`) — partially safe

- `run_subsentence_qe_orchestrate_*.sbatch` has `#SBATCH --requeue` (good).
- Explicit resume tooling exists: `resume_subsentence_qe_ja.sh`; `stage4_final_de.sh` has
  skip/exists logic. Stage1 LLM segmentation shards show **no skip-existing marker** —
  a preempted stage-1 shard reruns fully and overwrites its own shard file (idempotent
  output path, so no corruption, but wasted GPU hours).
- The known failure mode is **`afterok` chains**: a preempted array task → `CANCELLED` →
  downstream `DependencyNeverSatisfied`. Documented recovery: rerun the shard and
  `scontrol update jobid=<finalize> dependency=afterok:<rerun>`.

### Evaluation — effectively idempotent

`eval_all_ckpts*.sh` writes into per-ckpt `evaluation/acl_6060/<lang>/seg<N>/` dirs;
reruns overwrite the same instance/score files (safe to repeat, no accumulation bug seen).

## B. Design: `tools/babel/` for Mac-as-controller (design only — nothing replaced)

> **2026-08-28 update:** with the code checkout in `/home/haolingp/CMU_research_SMT`
> (`HOME_CHECKOUT.md`), `sync`/`submit`/`status`/`logs` all run directly on the login node
> (`git pull`, `sbatch`, `squeue`, `tail`); `ondata` is only needed for reading `/data` files.
> Partition/QOS available to you is now preempt only → `--requeue` + resume becomes mandatory.

Architecture: Claude/Codex on Mac edits the local repo → sync to Babel → sbatch on a
login node → results read back. Constraints that shape it: `/data/user_data` is invisible
on login nodes (use `~/bin/ondata`), and `sbatch`/`squeue` work fine on login nodes.

Proposed scripts (thin wrappers over ssh; create later under `tools/babel/`):

- **`sync.sh`** — push local changes up. Preferred path is git
  (`git push` → `ssh babel "~/bin/ondata 'git -C /data/user_data/haolingp pull --ff-only'"`),
  with an rsync fallback for untracked scratch:
  `rsync -az --exclude-from=data_synthesis/migration/exclude.txt ./ babel:migration_staging/up/`
  then `ondata 'rsync -a ~/migration_staging/up/ /data/user_data/haolingp/'`. Never `--delete`.
- **`submit.sh <script> [args]`** — `ssh babel "cd <script-dir-on-babel> && mkdir -p slurm_logs && sbatch <script>"`;
  echo the JobId; enforce the `--time ≤ partition MaxTime` rule by grepping the header
  before submitting (the #1 footgun).
- **`status.sh`** — `ssh babel 'squeue -u haolingp -o "%.10i %.9P %.28j %.8T %.10M %.6D %R"'`
  plus `sacct` for recently finished jobs.
- **`logs.sh <jobid>`** — tail `slurm_logs/<jobid>.{out,err}`; needs `ondata` when logs
  live under `/data/user_data` (most infer logs) but plain ssh when under `/home`.
- Keep a standing **anchor job** on the `cpu` partition so `ondata` always has a target
  (already your practice).

Guardrails for the wrappers: never auto-`scancel`; never submit train scripts whose
`EXP_NAME` save dir already contains checkpoints (guards issue A.2); refuse `--partition`
overrides that don't also override `--time`.
