---
title: Babel HPC Cluster (SLURM)
type: entity
tags: [infra, slurm, cluster, hpc, jobs]
sources:
  - https://wiki.babel.cs.cmu.edu/index.php/BABEL
  - live config — sinfo / scontrol show partition / sacctmgr show qos (2026-06-20)
created: 2026-06-20
updated: 2026-06-20
---

# Babel HPC Cluster (SLURM)

The CMU LTI cluster everything in this project runs on. SLURM **24.05.1**. Login:
`ssh <andrew_id>@login.babel.cs.cmu.edu` (Andrew ID, **not** SCS creds; 4 round-robin login
nodes — `ssh login2` to reach a specific one). All training/synthesis/eval jobs here are `sbatch`
(see [[megatron-swift]], [[synthesis-pipeline]], [[streaming-inference]], [[checkpoint-evaluation]]).

## Partitions — MaxTime is a HARD ceiling (the #1 footgun)

| Partition | MaxTime | DefaultTime | MaxGPU | MaxCPU | QoS | Notes |
|---|---|---|---|---|---|---|
| `general`* | **2-00:00:00** (2 d) | 6 h | 8 | 128 | `normal` | default; sbatch only, no interactive |
| `preempt` | 31-00:00:00 | 3 h | 24 | 256 | `preempt_qos` | **jobs get evicted anytime**; sbatch only |
| `cpu` | 2-00:00:00 | 6 h | 0 | 128 | `cpu_qos` | CPU-only; sbatch only |
| `array` | 12-00:00:00 | 6 h | 8 | 256 | `array_qos` | array jobs; sbatch only |
| `debug` | 12:00:00 | 1 h | 2 | 64 | `debug_qos` | **no array jobs**; interactive OK |

**If `--time` > the partition MaxTime, the job is parked forever as `(PartitionTimeLimit)` and
never schedules** — it is NOT clamped. A script written for `preempt` (`--time=3-00:00:00`)
submitted to `general` (max 2 d) hangs silently. **Always set `--time` ≤ partition MaxTime**, and
when overriding a preempt script's partition, override `--time` too. (This exact mistake stalled
the `top5-axis5-qwenasr` QE rerun — fixed with `scontrol update jobid=<id> TimeLimit=01:00:00`.)

## QoS limits (per user)

| QoS | MaxGPU (TRESPU) | MaxJobsPU (running) | MaxSubmitPU | MaxCPU | preemptable by |
|---|---|---|---|---|---|
| `normal` | 8 | 10 | 50 | 128 | array_qos, preempt_qos |
| `preempt_qos` | 24 | 24 | 100 | 256 | — |
| `cpu_qos` | 0 | 10 | 50 | — | preempt_qos |
| `array_qos` | 8 | 100 | 10000 | 256 | preempt_qos |
| `debug_qos` | 2 | 10 | 12 | 64 | preempt_qos |

`MaxSubmitPU` caps queued+running; large `afterok` fan-outs can hit it. Lab QoS also exist
(`maxlab_qos` gpu=16, `rl_qos` gpu=32, `dlab_qos` gpu=24, …) via the right `--account`.

## GPU request syntax

`--gres=gpu:<type>:<n>`. Types present: **L40S** (most common here, 46 GB), L40, A6000, 6000Ada,
RTX_PRO_6000, A100_40GB, A100_80GB, H200, H100. This project's runs use `gpu:L40S` (train 4×,
synthesis/decode 2×, QE/MetricX 1× per shard).

## Storage / quota (the AutoFS gotcha)

| Path | Size | Scope |
|---|---|---|
| `/home/<user>` | **100 GB** | all nodes |
| `/data/user_data/<user>` | **500 GB** | **only compute nodes with an active job** |
| `/data/datasets`, `/data/models` | community | compute nodes |
| `/scratch` | local SSD/NVMe | per-node; **files >28 d auto-expunged when >65 % full** |
| `/compute/<node>` | — | each node's scratch, exported cluster-wide |
| `/data/group_data/<group>` | per-group | group share (e.g. `li_lab/siqiouya/datasets/gigaspeech`) |

**AutoFS = on-demand mount: you must `stat` the full path to trigger the mount**, else the dir
looks empty / `df` reports nonsense. This explains the transient "Disk quota exceeded / avail=0"
seen mid-session that later showed 2.8 TB free — and why `/data/user_data` is invisible off-node.
The frozen-reference TSVs and consensus outputs live under `/data/user_data/haolingp/...`; the
source manifests under `/data/group_data/li_lab/siqiouya/...` ([[gigaspeech]]).

## Job-submission rules of thumb (learned here)

1. **`--time` ≤ partition MaxTime**, every time. Overriding `--partition` ⇒ override `--time`.
2. **preempt is for long, requeue-safe jobs** (`--requeue` + skip-existing/sentinel resume). For
   short, critical, afterok-gated steps (a finalize, a merge, a few-minute QE shard), prefer
   `general`/`cpu` so an eviction doesn't break the dependency chain.
3. **`afterok` is unforgiving**: if any array task ends non-`COMPLETED` (preemption → CANCELLED),
   the dependent job goes `DependencyNeverSatisfied` and the whole downstream chain stalls. Rerun
   the failed shards, then `scontrol update jobid=<finalize> dependency=afterok:<rerun>` to release
   it (re-pointing keeps the rest of the chain intact).
4. **Unique ports per co-located task**: base server ports on `SLURM_ARRAY_TASK_ID` (spaced ≥2),
   not `JOB_ID%100` — adjacent array tasks sharing a node otherwise collide (bit the J40k decode).
5. No interactive sessions on general/preempt/cpu/array — use `debug` (`srun`/`salloc`); no array
   jobs on `debug`.

## Sources
- [[megatron-swift]], [[synthesis-pipeline]], [[streaming-inference]], [[checkpoint-evaluation]],
  [[metricx]], [[gigaspeech]]
- code: every `*.sbatch` / `submit_*.sh` under `../codes/` and `scripts/`
