# CMU_research_SMT — simultaneous speech translation (data synthesis → training → eval)

> **Quick start / operations guide (Babel + Mac controller, updated 2026-08-28).**
> The original pipeline description follows below under *Streaming Translation Data Synthesis Pipeline*.
> Deeper docs: `data_synthesis/migration/` (`HOME_CHECKOUT.md`, `PROJECT_WORKFLOW.md`,
> `BABEL_PATHS.md`, `PREEMPTION_READINESS.md`, `PULL_TO_MAC.md`) and the LLM wiki at `data_synthesis/wiki/`.

## 1. Where things live

| What | Path | Visible from |
|---|---|---|
| **Code (this repo, canonical)** | `/home/haolingp/CMU_research_SMT` on Babel | login + compute nodes |
| Local copy on Mac | `/Users/haolingpu/Desktop/research/CMU_research_SMT` | Mac |
| Synthesized data | `/data/user_data/haolingp/data_synthesis/outputs/` (1.8 TB) | compute nodes only |
| Checkpoints | `/data/user_data/haolingp/ckpts/infinisst-omni/<EXP>/` (3.1 TB) | compute nodes only |
| Model weights / HF cache | `/data/user_data/haolingp/{models,hf_cache}/` | compute nodes only |
| GigaSpeech source | `/data/group_data/li_lab/siqiouya/datasets/gigaspeech/` | compute nodes only |
| External deps (Megatron-LM, OmniSTEval, SEGALE w/ local patch, metricx) | `/data/user_data/haolingp/{code,tools}/` — pinned + patch in `external/` | compute nodes only |
| Secrets | `/data/user_data/haolingp/.env`, `~/.keys/{wandb,huggingface}` — never commit | — |
| Job logs | `scripts/{train,infer}/slurm_logs/`, `data_synthesis/codes/**/slurm_logs/` (in this repo → readable from login node) | login + compute |

Login nodes do **not** mount `/data/user_data`. To read data files from a login node use
`~/bin/ondata '<command>'` (hops into one of your running jobs; keep an `anchor` job alive:
`sbatch -p preempt -q preempt_cpu_qos --requeue -t 2-00:00:00 -J anchor --wrap 'sleep infinity'`).

## 2. Environment

Conda envs on Babel (`source ~/miniconda3/etc/profile.d/conda.sh && conda activate <env>`):

| Env | Used for |
|---|---|
| `evaluation` | simuleval / omnisteval streaming eval (torch 2.10 cu128) |
| `vllm` | LLM synthesis servers (vLLM) |
| `metricx` | MetricX-24 QE filtering |
| `segale` | SEGALE QE — **sm_89 only** (L40S / 6000Ada); never run on RTX_PRO_6000 |
| `SMT`, `gemma4` | misc synthesis / gemma experiments |

Training runs inside **apptainer** (`scripts/train/launch_container.sh`, ms-swift + Megatron image);
no conda env needed. Text captures of every env (pip freeze, conda list, key versions) are in
`data_synthesis/migration/environment/`. Nothing here runs on the Mac except analysis/plots.

## 3. Pipeline & entrypoints

```
synthesis  data_synthesis/codes/gigaspeech/{east,refined_east,salami,future_sampling}/
           submit_*.sh → stage1 (LLM segmentation) → stage2 (streaming) → stage3 (MetricX QE) → stage4 (final JSONL)
convert    scripts/train/convert2swift_*.py  (+ run_convert2swift_*.sbatch)  → SWIFT manifest
train      scripts/train/train_<EXP>_s.sh     (4×L40S, Qwen3-Omni-30B LoRA via Megatron-SWIFT in apptainer)
eval       scripts/infer/eval_all_ckpts*.sh   reads scripts/infer/ckpts*.txt → infer_slurm*.sh → omnisteval
                                              results land in ckpts/infinisst-omni/<EXP>/v*-hf/evaluation/
```
Gotchas: zh EAST/Simul models were trained with the *Standard* prompt → infer with
`--prompt-type Standard`; streaming `max-new-tokens` is ja=30 / de=60 per chunk; synthesis-time
BLEU does not predict trained-model BLEU — rank by COMET.

## 4. Running jobs (from a Babel login node)

```bash
cd ~/CMU_research_SMT && git pull --ff-only
sbatch scripts/train/train_EAST-latency2mult_s.sh          # training
sbatch data_synthesis/codes/gigaspeech/east/submit_de.sh   # synthesis (or bash, depending on script)
bash   scripts/infer/eval_all_ckpts.sh                      # eval fan-out
squeue -u haolingp                                          # status
tail -f scripts/train/slurm_logs/<job>.out                  # logs (in /home, no ondata needed)
```

SLURM rules for this account (as of 2026-08-28): **only `preempt` partition / `preempt_qos`
& `preempt_cpu_qos`** are available. Scripts that still say `--partition=general` must be
submitted with `sbatch -p preempt -q preempt_qos ...` (CLI overrides `#SBATCH`), and
`--time` must be ≤ partition MaxTime. Keep resource requests lean (4 CPU / 32 GB for vLLM
serves; 4×L40S for training). Preempted training currently **restarts from step 0**
(`--finetune true`, no `--requeue`) — see `PREEMPTION_READINESS.md` before long runs.

## 5. Mac ↔ Babel workflow

```bash
# Mac: edit, then
git push
ssh babel 'cd ~/CMU_research_SMT && git pull --ff-only && sbatch <script>'
ssh babel 'squeue -u haolingp'
ssh babel 'tail -50 ~/CMU_research_SMT/scripts/train/slurm_logs/<job>.out'
ssh babel "~/bin/ondata 'ls /data/user_data/haolingp/ckpts/infinisst-omni/<EXP>'"   # /data files only
```
Never rsync `.git` from `/data/user_data/haolingp` (966 GB of orphaned pack files); that tree
is legacy — data only. Initial Mac setup is in `data_synthesis/migration/PULL_TO_MAC.md`.

---

# Streaming Translation Data Synthesis Pipeline

This repository builds high-quality **bilingual streaming translation segments**
from raw audio-text corpora, supporting simultaneous translation and low-latency
MT research.

Two corpora are supported:

| Corpus | Code Directory | Data Source |
|---|---|---|
| **YODAS** | `data_synthesis/codes/yodas/` | YODAS-Granary Parquet files |
| **GigaSpeech** | `data_synthesis/codes/gigaspeech/` | GigaSpeech manifests (TSV) |

Both share the same pipeline architecture, with three segmentation methods:
**EAST** (baseline), **Refined EAST**, and **SALAMI**.

------------------------------------------------------------
## Repository Structure
------------------------------------------------------------

```
data_synthesis/codes/
├── yodas/                          # YODAS corpus pipeline
│   ├── llm_output_vllm_batch.py    # LLM segmentation (vLLM, batched)
│   ├── llm_output_vllm.py          # LLM segmentation (vLLM, single)
│   ├── llm_output.py               # LLM segmentation (API-based)
│   ├── llm_output_Salami.py        # SALAMI-format LLM segmentation
│   ├── post_process_llm_output.py  # Merge single-word chunks
│   ├── find_bad_json.py            # Quality filter (LLM vs MFA alignment check)
│   ├── multi_trajectory.py         # Build streaming trajectories (960ms chunks)
│   ├── export_mfa_corpus.py        # Export audio+text for MFA alignment
│   ├── convert_metricx.py          # Convert streaming output → MetricX QE input
│   ├── filter_metricx.py           # Filter by MetricX QE score
│   ├── final_output.py             # Build final JSONL dataset
│   ├── fix_source_punct.py         # Post-fix: restore punctuation from manifest
│   ├── gen_manifest_report.py      # Validate final output against manifest
│   ├── evaluate_3_methods.py       # Compare EAST / Refined EAST / SALAMI outputs
│   ├── check_manifest_match.py     # Check manifest token alignment
│   ├── compare_metricx.py          # Compare MetricX scores across methods
│   ├── validate_final_output.py    # Validate final dataset integrity
│   ├── build_streaming_segments.py # Single-utterance demo (for debugging)
│   │
│   ├── run_segmentation_array.sh   # LLM segmentation (8-GPU SLURM array)
│   ├── run_EAST_baseline.sh        # Full EAST pipeline (SLURM)
│   ├── run_full_pipeline.sh        # Full Refined EAST pipeline (SLURM)
│   ├── run_salami.sh               # Full SALAMI pipeline (SLURM)
│   ├── pipeline.sh                 # MetricX scoring + filtering (SLURM)
│   ├── start_vllm_dp.sh            # Start vLLM server (data parallel)
│   └── ...
│
├── gigaspeech/                     # GigaSpeech corpus pipeline
│   ├── fix_llm_raw.py              # Fix LLM raw output (restore punct, filter mismatches)
│   ├── post_process_llm_output_gigaspeech.py
│   ├── find_bad_json_gigaspeech.py
│   ├── multi_trajectory_gigaspeech.py
│   ├── convert_metricx_gigaspeech.py
│   ├── filter_metricx_gigaspeech.py
│   ├── final_output_gigaspeech.py
│   ├── export_mfa_corpus_gigaspeech.py
│   ├── llm_output_gigaspeech_trajectory.py
│   ├── check_gigaspeech_manifest.py
│   │
│   ├── east/                       # EAST method scripts
│   ├── refined_east/               # Refined EAST method scripts
│   └── salami/                     # SALAMI method scripts
│
└── metricx/                        # MetricX-24 (submodule)
```

------------------------------------------------------------
## Pipeline Overview
------------------------------------------------------------

All three methods (EAST / Refined EAST / SALAMI) follow the same stages:

```
[1] LLM Segmentation        (vllm env, multi-GPU)
         ↓
[2] Fix LLM Output          (restore punctuation from manifest, discard mismatches)
         ↓
[3] Post-process             (merge single-word chunks)
         ↓
[4] MFA Forced Alignment     (export corpus → MFA align → TextGrid)
         ↓
[5] Quality Filter           (find_bad_json: check LLM ↔ MFA token alignment)
         ↓
[6] Build Streaming Dataset  (multi_trajectory: 960ms chunk-based timeline)
         ↓
[7] MetricX QE Scoring       (metricx env, per-segment quality estimation)
         ↓
[8] Threshold Filtering      (filter by MetricX score ≤ τ)
         ↓
[9] Final Dataset            (JSONL with source/target per latency level)
         ↓
[10] Fix Source Punctuation   (optional: restore punct in final output)
```

------------------------------------------------------------
## Data Sources
------------------------------------------------------------

### YODAS
- **Parquet files**: Raw audio bytes + ASR transcripts (from YODAS-Granary)
- **MFA output**: Word-level TextGrid per utterance
- **LLM segmentation JSONs**: Low/medium/high latency English↔Chinese chunks

### GigaSpeech
- **Manifest TSV**: `/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv`
- Same MFA + LLM pipeline structure as YODAS

------------------------------------------------------------
## Key Pipeline Steps
------------------------------------------------------------

### 1. LLM Segmentation

Segment transcripts into bilingual (English→Chinese) chunks at multiple latency
levels using vLLM (Qwen3-30B or similar).

```bash
# YODAS: 8-GPU array job
sbatch data_synthesis/codes/yodas/run_segmentation_array.sh

# GigaSpeech
sbatch data_synthesis/codes/gigaspeech/east/llm.sh
```

### 2. Fix LLM Output (GigaSpeech)

Restore punctuation from the original manifest into LLM output, discarding
utterances with token mismatches. This step is critical for GigaSpeech where
the LLM may strip or alter punctuation.

```bash
python fix_llm_raw.py \
  --in_dir  .../llm_output_raw \
  --out_dir .../llm_output_raw_fixed \
  --good_jsonl .../good.jsonl \
  --out_good_jsonl .../good_fixed.jsonl
```

### 3. Post-process & Quality Filter

Merge single-word chunks, then check LLM ↔ MFA alignment:

```bash
python post_process_llm_output.py --input_dir ... --output_dir ... --lang en000
python find_bad_json.py --llm-root ... --mfa-root ... --corpus-root ... --lang en000
```

### 4. Build Streaming Dataset

Construct per-second emission timeline using MFA word timestamps:

```bash
python multi_trajectory.py \
  --llm-root .../llm_output_merged \
  --mfa-root .../textgrids \
  --good-root .../good.jsonl \
  --output-root .../streaming_dataset \
  --langs en000
```

### 5. MetricX QE Scoring & Filtering

Score each segment with MetricX-24-Hybrid-XL (QE mode), then filter:

```bash
python convert_metricx.py --stream_dir ... --output metricx_input.jsonl

PYTHONNOUSERSITE=1 python -m metricx24.predict \
  --tokenizer .../mt5-xl \
  --model_name_or_path .../metricx-24-hybrid-xl-v2p6 \
  --input_file metricx_input.jsonl \
  --output_file metricx_output.jsonl --qe

python filter_metricx.py --input metricx_output.jsonl --output filtered.jsonl --threshold 3.0
```

### 6. Final Dataset & Post-fix

```bash
python final_output.py --metricx_jsonl filtered.jsonl --stream_dir ... --output_dir ...

# Optional: restore punctuation in final output
python fix_source_punct.py --final_dir ... --stream_dir ... --out_dir ... --out_flat good_fixed.jsonl
```

------------------------------------------------------------
## Output Format
------------------------------------------------------------

Each utterance produces a JSON with bilingual streaming segments:

```json
{
  "utt_id": "utt_en000_00000000_0003",
  "original_text": "...",
  "source_low_latency": ["which sites", "have you been", "using"],
  "target_low_latency": ["哪些网站", "你一直在", "使用"],
  "source_medium_latency": [...],
  "target_medium_latency": [...],
  "source_high_latency": [...],
  "target_high_latency": [...]
}
```

------------------------------------------------------------
## Environments
------------------------------------------------------------

| Env | Purpose |
|---|---|
| `vllm` | LLM inference (vLLM + Qwen3) |
| `SMT` | Pipeline processing (MFA, data manipulation) |
| `metricx` | MetricX-24 QE scoring (Python 3.9, separate deps) |

------------------------------------------------------------
## Evaluation & Utilities
------------------------------------------------------------

| Script | Purpose |
|---|---|
| `evaluate_3_methods.py` | Side-by-side comparison of EAST / Refined EAST / SALAMI |
| `compare_metricx.py` | Compare MetricX score distributions across methods |
| `gen_manifest_report.py` | Validate final output against original manifest text |
| `check_manifest_match.py` | Check token-level alignment with manifest |
| `validate_final_output.py` | Validate dataset integrity |

------------------------------------------------------------
## Running Full Pipelines
------------------------------------------------------------

### YODAS

```bash
# EAST baseline
sbatch data_synthesis/codes/yodas/run_EAST_baseline.sh

# Refined EAST
sbatch data_synthesis/codes/yodas/run_full_pipeline.sh

# SALAMI
sbatch data_synthesis/codes/yodas/run_salami.sh
```

### GigaSpeech

```bash
# Step 0: MFA corpus (one-time, shared)
sbatch data_synthesis/codes/gigaspeech/run_train_xl_corpus_array_8gpu.sh

# Step 1: LLM segmentation (pick one method)
sbatch data_synthesis/codes/gigaspeech/east/llm.sh
sbatch data_synthesis/codes/gigaspeech/refined_east/llm.sh
sbatch data_synthesis/codes/gigaspeech/salami/llm.sh

# Step 2: Pipeline (fix → post-process → filter → trajectory → metricx prep)
sbatch data_synthesis/codes/gigaspeech/east/pipeline.sh
sbatch data_synthesis/codes/gigaspeech/refined_east/pipeline.sh
sbatch data_synthesis/codes/gigaspeech/salami/pipeline.sh

# Step 3: MetricX predict (8-shard parallel)
sbatch data_synthesis/codes/gigaspeech/east/run_metricx_predict.sh

# Step 4: Merge shards → filter → final dataset
```

See `data_synthesis/codes/gigaspeech/README.md` for detailed GigaSpeech instructions.
