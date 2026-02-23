# GigaSpeech Data Synthesis Pipeline

## Directory Structure

```
gigaspeech/
├── README.md
├── fix_llm_raw.py                        # Restore punctuation from manifest, filter token mismatches
├── post_process_llm_output_gigaspeech.py  # Merge one-word chunks
├── multi_trajectory_gigaspeech.py         # Build streaming trajectories (960ms chunks)
├── find_bad_json_gigaspeech.py            # Quality filter (MFA alignment check)
├── convert_metricx_gigaspeech.py          # Convert to MetricX QE input format
├── filter_metricx_gigaspeech.py           # Filter by MetricX QE score
├── final_output_gigaspeech.py             # Build final JSONL dataset
├── export_mfa_corpus_gigaspeech.py        # Export audio+text for MFA alignment
├── llm_output_gigaspeech_trajectory.py    # LLM segmentation (EAST format)
├── check_gigaspeech_manifest.py           # Manifest validation utility
├── run_train_xl_corpus_array_8gpu.sh      # MFA corpus export (8-GPU array job, shared)
│
├── east/                    # EAST baseline method
│   ├── pipeline.sh          # Full pipeline: fix → post-process → find_bad → trajectory → metricx
│   ├── llm.sh               # LLM segmentation (8-GPU array job)
│   ├── run_metricx_predict.sh
│   └── run_dev_llm_8gpu.sh  # Dev/test LLM job
│
├── refined_east/            # Refined EAST method
│   ├── pipeline.sh
│   ├── llm.sh
│   └── run_metricx_predict.sh
│
└── salami/                  # SALAMI method
    ├── pipeline.sh          # Includes extra salami→offline mapping step
    ├── llm.sh
    ├── run_metricx_predict.sh
    ├── llm_output_salami.py               # SALAMI-format LLM output generator
    └── map_salami_to_offline_gigaspeech.py # Convert salami format → offline format
```

## Prerequisites

- **Conda environments:** `vllm` (LLM inference), `SMT` (pipeline processing), `metricx` (quality scoring)
- **Manifest:** `/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv`
- **Models:** `mt5-xl` tokenizer, `metricx-24-hybrid-xl-v2p6`

## Pipeline Overview

Each method (EAST / Refined EAST / SALAMI) follows the same stages:

```
[1] LLM Segmentation  (vllm env, 8 GPUs)
         ↓
[2] Pipeline           (SMT + metricx envs, 1 GPU)
    ├── fix_llm_raw
    ├── post_process (merge one-word chunks)
    ├── find_bad_json (quality filter)
    ├── multi_trajectory (build streaming dataset)
    ├── convert_metricx (prepare MetricX input)
    └── split into 8 shards
         ↓
[3] MetricX Predict    (metricx env, 8 GPUs array)
         ↓
[4] Post-MetricX       (merge shards → filter → final dataset)
```

## How to Run

### Step 0: MFA Corpus (one-time, shared across methods)

```bash
sbatch run_train_xl_corpus_array_8gpu.sh
```

### Step 1: LLM Segmentation

```bash
# Pick one:
sbatch east/llm.sh
sbatch refined_east/llm.sh
sbatch salami/llm.sh
```

### Step 2: Pipeline (fix + post-process + trajectory + metricx prep)

```bash
# Pick one:
sbatch east/pipeline.sh
sbatch refined_east/pipeline.sh
sbatch salami/pipeline.sh
```

This runs through metricx input split. Wait for completion.

### Step 3: MetricX Predict (8 shards in parallel)

```bash
# Pick one:
sbatch east/run_metricx_predict.sh
sbatch refined_east/run_metricx_predict.sh
sbatch salami/run_metricx_predict.sh
```

Wait for all 8 array tasks to complete.

### Step 4: Merge + Filter + Final Dataset

After MetricX predict finishes, merge shards and run filtering. The metricx predict scripts include commented-out commands for this, or run manually:

```bash
# Example for EAST:
OUT=/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east

# Merge shards
cat ${OUT}/metricx_shards/output_*.jsonl > ${OUT}/metricx_output.jsonl

# Filter
conda activate metricx
python filter_metricx_gigaspeech.py \
  --input ${OUT}/metricx_output.jsonl \
  --output ${OUT}/metricx_filtered_t3.0.jsonl \
  --threshold 3.0

# Final dataset
python final_output_gigaspeech.py \
  --metricx_jsonl ${OUT}/metricx_filtered_t3.0.jsonl \
  --stream_dir ${OUT}/streaming_EAST_dataset \
  --output_dir ${OUT}/final_jsonl_dataset
```

## Output Directories

All outputs are under `/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/`:

| Method | Output Dir |
|---|---|
| EAST | `train_xl_east/` |
| Refined EAST | `train_xl_refined_east/` |
| SALAMI | `train_xl_salami/` |
| MFA (shared) | `train_xl_east/mfa_corpus/`, `train_xl_east/mfa_textgrid/` |

## Key Differences Between Methods

| | EAST | Refined EAST | SALAMI |
|---|---|---|---|
| LLM format | low/medium/high latency | low/medium/high latency | segmented_pairs |
| Extra step | - | - | salami → offline mapping |
| find_bad flag | default | default | `--allow-one-word` |
| LLM script | shared `llm_output_gigaspeech_trajectory.py` | shared | `salami/llm_output_salami.py` |
