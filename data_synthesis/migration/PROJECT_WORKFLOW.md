# Project Workflow (verified against scripts, 2026-08-24)

Simultaneous speech-translation research: offline **data synthesis** (EAST / SALAMI /
future-sampling / consensus policies) from GigaSpeech → **LoRA training** of Qwen3-Omni via
SWIFT/Megatron → **streaming evaluation** on ACL 6060 dev with simuleval + omnisteval.

The `data_synthesis/wiki/` LLM Wiki is the authoritative deep documentation
(`wiki/index.md` is the catalog; `wiki/WIKI.md` the conventions). This file is the
operational quick reference.

```
SYNTHESIS (GigaSpeech manifests → streaming trajectories, MetricX-QE filtered)
    ↓  JSONL per recording/latency, under data_synthesis/outputs/
CONVERSION (convert2swift_*.py → SWIFT training manifests, group_data)
    ↓
TRAINING (SWIFT Megatron LoRA in apptainer container, 4×L40S, general partition)
    ↓  ckpts/infinisst-omni/<EXP_NAME>/v*-hf/
EVALUATION (simuleval streaming infer → normalize → omnisteval → scores.tsv)
    ↓
ANALYSIS (scoreboard in wiki/comparisons/scoreboard.md, plot scripts)
```

## 1. Synthesis / trajectory generation

Working dir: `data_synthesis/codes/gigaspeech/` (per-method subdirs: `east/`, `salami/`,
`refined_east/`, `future_sampling/`, `rule-based-SMT/`, `hibiki/`).

- Orchestrator per method: `east/pipeline.sh`, `salami/pipeline.sh`; newer EAST de/ja runs
  use staged scripts `east/stage1_{de,ja}.sh` → `stage3_metricx_*.sh` → `stage4_final_*.sh`
  driven by `east/submit_{de,ja}*.sh` (sbatch chains with afterok deps).
- Stage 1: LLM segmentation — `llm_output_gigaspeech_trajectory.py` (8-GPU array, `vllm` env).
- Stage 2: post-processing — `fix_llm_raw.py` → `post_process_llm_output_gigaspeech.py` →
  `find_bad_json_gigaspeech.py` (MFA check) → `multi_trajectory_gigaspeech.py` (~960 ms chunks)
  → `convert_metricx_gigaspeech.py`.
- Stage 3: MetricX QE (8-GPU array, `metricx` env; model `models/metricx-24-hybrid-xl-v2p6`).
- Stage 4: `filter_metricx_gigaspeech.py` (keep ≤ 3.0) → `final_output_gigaspeech.py`.
- Consensus decoding (current research focus): `data_synthesis/codes-refactored/consensus_decoding/`
  + `consensus_decoding_token_id_level.py`; sbatch in `scripts/synth/run_consensus.sbatch`.
- Rule-based offline policies (LA-N, wait-k, prefix-alignment):
  `codes/gigaspeech/rule-based-SMT/`.

Outputs → `data_synthesis/outputs/{EAST,Refined_EAST,SALAMI,gigaspeech}/…` (Babel-only).

## 2. Dataset conversion

`scripts/train/convert2swift_*.py` (one per method/language: EAST, LA, PA, consensus,
hibiki, salami_{de,ja}, simul-mustc, wordalign, …), submitted via
`scripts/train/run_convert2swift_*.sbatch`. Produces SWIFT manifests under
`/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/` (audio paths inside are
group-readable). See wiki `concepts/dataset-conversion-pipeline.md`.

## 3. Training

- Scripts: `scripts/train/train_*.sh` (~23 variants, one per experiment); sbatch headers:
  4×L40S, `--partition=general`, `--time=1-00:00:00`, 16 CPU, 128 GB.
- Runtime: **apptainer** modelscope image via `scripts/train/launch_container.sh`; SWIFT
  Megatron CLI, LoRA r32/α32, mbs1/gbs4, LR 1e-4, 1 epoch, `--save_interval 200`.
- Secrets: reads `/home/haolingp/.keys/{wandb,huggingface}` at submit time (never copy).
- Checkpoints → `/data/user_data/haolingp/ckpts/infinisst-omni/<EXP_NAME>/`; HF export
  `v<N>-…-hf/`; mcore conversion via `convert_qwen3omni_to_mcore.sh`.
- Chained train→infer: `launch_infer_after_train*.sbatch`, `hibiki_chain.sh`,
  `run_win3_*chain.sh` (afterok chains).

## 4. Evaluation

- Driver: `scripts/infer/eval_all_ckpts{,_ja,_de,_v2,_simultst}.sh`; checkpoint registry
  files `scripts/infer/ckpts*.txt` (ckpts.txt, ckpts_de.txt, ckpts_ja.txt, …).
- Per checkpoint × 4 segment sizes: streaming inference (`infer_slurm*.sh`, agent
  `scripts/infer/infinisst_omni.py`, simuleval + vLLM, `evaluation` env) →
  `instances.log` → `normalize_instances.py` (NFKC) → **omnisteval longform**
  (`code/OmniSTEval`) → `scores.tsv`.
- Metrics: BLEU (char-level zh/ja), COMET `Unbabel/XCOMET-XL`, chrF, LongYAAL latency.
- Test set: ACL 6060 dev (`datasets/acl_6060`); monotonic-ref set in `simul_tst_common/`.
- Results land in `ckpts/infinisst-omni/<exp>/v*-hf/evaluation/acl_6060/<lang>/seg<N>/`;
  consolidated scoreboard: `data_synthesis/wiki/comparisons/scoreboard.md`.
- **Gotcha**: zh EAST-latency2mult & Simul-MuST-C were trained with the Standard prompt —
  infer with `--prompt-type Standard` (wiki `concepts/east-prompt-handling.md`).
- Plots: `scripts/infer/plot_latency_quality_*.py`, `scripts/debug/`.

## 5. Environments & activation

| Env | Used for | Location |
|---|---|---|
| `evaluation` | simuleval/eval (most-used) | `/data/user_data/haolingp/conda_envs/evaluation` |
| `vllm` | LLM segmentation/serving | `~/miniconda3/envs/vllm` |
| `metricx` | QE scoring | `~/miniconda3/envs/metricx` |
| `SMT` | rule-based policies etc. | `~/miniconda3/envs/SMT` |
| `segale` | SEGALE alignment/QE (**sm_89 only — pin `--gres=gpu:L40S`**) | `conda_envs/segale` |
| (training) | apptainer container, not conda | `launch_container.sh` |

Activation: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate <env>`.
Full version dumps: `migration/environment/`.

## 6. SLURM operational rules (full detail: wiki `entities/babel-cluster.md`)

- `--time` ≤ partition MaxTime or the job parks forever (general/cpu 2 d, debug 12 h,
  preempt 31 d); overriding `--partition` ⇒ override `--time`.
- preempt = long requeue-safe jobs only; keep short afterok-gated steps on general/cpu;
  re-point stalled dependencies with `scontrol update jobid=… dependency=afterok:…`.
- Lean requests: 4 CPU/32 GB even for vLLM serves; L40S most available.
- Expected working dir for most sbatch scripts: the script's own directory (they use
  relative `slurm_logs/%j.out` paths — a `slurm_logs/` dir must exist in cwd).
