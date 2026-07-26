#!/usr/bin/env bash
#SBATCH --job-name=hibiki_argmax100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=120G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=1-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/codes/gigaspeech/hibiki/output/argmax/100cases/slurm_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/codes/gigaspeech/hibiki/output/argmax/100cases/slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

source ~/.bashrc
conda activate vllm

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIBIKI_DIR="$(cd "$CODE_DIR/.." && pwd)"
PYTHON_SCRIPT="$CODE_DIR/contextual_alignment.py"
MODEL_PATH="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
INPUT_TSV="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv"
OUTPUT_DIR="$HIBIKI_DIR/output/argmax/100cases"

MAX_ROWS=100
TP=1
GPU_MEM_UTIL=0.90

mkdir -p "$OUTPUT_DIR"

echo "job_id=${SLURM_JOB_ID:-N/A} host=$(hostname) time=$(date)"
echo "script=$PYTHON_SCRIPT"
echo "model=$MODEL_PATH"
echo "input=$INPUT_TSV"
echo "max_rows=$MAX_ROWS"
echo "output_dir=$OUTPUT_DIR"

python "$PYTHON_SCRIPT"   --input-tsv "$INPUT_TSV"   --tokenizer-path "$MODEL_PATH"   --base-model-path "$MODEL_PATH"   --max-rows "$MAX_ROWS"   --tp "$TP"   --gpu-memory-utilization "$GPU_MEM_UTIL"   --output-jsonl "$OUTPUT_DIR/contextual_alignment_argmax_100.jsonl"   --output-txt "$OUTPUT_DIR/contextual_alignment_argmax_100.txt"   --output-pretty-json "$OUTPUT_DIR/contextual_alignment_argmax_100.pretty.json"


echo "done time=$(date)"
echo "pretty_json=$OUTPUT_DIR/contextual_alignment_argmax_100.pretty.json"
echo "txt=$OUTPUT_DIR/contextual_alignment_argmax_100.txt"
echo "jsonl=$OUTPUT_DIR/contextual_alignment_argmax_100.jsonl"
