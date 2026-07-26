#!/usr/bin/env bash
# Debug run: token-level contextual alignment (Hibiki-style)
# Runs on 1-2 utterances with full verbose logging.
# Uses 1 GPU by default. Adjust --tp and CUDA_VISIBLE_DEVICES for multi-GPU.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIBIKI_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/data/user_data/haolingp/models}"

INPUT_TSV="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv"
MODEL_PATH="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
OUTPUT_DIR="$HIBIKI_DIR/output/debug_contextual_output"
mkdir -p "$OUTPUT_DIR"

MAX_ROWS=1  # Start with 1 utterance for debugging

echo "=== Hibiki contextual alignment debug run ==="
echo "  GPU:        $CUDA_VISIBLE_DEVICES"
echo "  Model:      $MODEL_PATH"
echo "  Input:      $INPUT_TSV"
echo "  Max rows:   $MAX_ROWS"
echo "  Output dir: $OUTPUT_DIR"
echo ""

python contextual_alignment_DP.py \
  --input-tsv "$INPUT_TSV" \
  --base-model-path "$MODEL_PATH" \
  --tokenizer-path "$MODEL_PATH" \
  --max-rows "$MAX_ROWS" \
  --tp 1 \
  --gpu-memory-utilization 0.90 \
  --output-jsonl "$OUTPUT_DIR/contextual_debug.jsonl" \
  --output-txt "$OUTPUT_DIR/contextual_debug.txt" \
  --output-pretty-json "$OUTPUT_DIR/contextual_debug.pretty.json" \
  2>&1 | tee "$OUTPUT_DIR/contextual_debug.log"

echo ""
echo "=== Done ==="
echo "  Log:         $OUTPUT_DIR/contextual_debug.log"
echo "  Pretty JSON: $OUTPUT_DIR/contextual_debug.pretty.json"
echo "  Readable:    $OUTPUT_DIR/contextual_debug.txt"
