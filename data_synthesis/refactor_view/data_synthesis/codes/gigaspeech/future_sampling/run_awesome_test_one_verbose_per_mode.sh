#!/usr/bin/env bash
# 对每个 selection-mode 用 awesome-align（原 core）各跑一条 test-one + verbose。
# 与 run_v2_test_one_verbose_per_mode.sh 结构相同，只是不用 simalign，用默认 awesome-align。
# Align 与 base 都在 GPU 0 上。不传参数则用下面默认 path 跑全部 mode。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INPUT_TSV="${1:-/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv}"
OUTPUT_ROOT="${2:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/out_awesome_test_one}"

if [[ ! -f "$INPUT_TSV" ]]; then
  echo "ERROR: input TSV not found: $INPUT_TSV"
  echo "Usage: $0 [input.tsv] [output_root]"
  exit 1
fi
mkdir -p "$OUTPUT_ROOT"

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${HF_HOME:-/data/user_data/haolingp/models}"

for MODE in majority_vote lcp70_llm lcp70_code lcp_code; do
  OUT="$OUTPUT_ROOT/awesome_verbose_${MODE}"
  mkdir -p "$OUT"
  echo "========== awesome-align mode=$MODE -> $OUT =========="
  python llm_future_sampling_core.py \
    --input-tsv "$INPUT_TSV" \
    --output-root "$OUT" \
    --test-one \
    --verbose \
    --selection-mode "$MODE" \
    --consensus-ratio "0.7" \
    --overwrite
  echo ""
done

echo "Done. Verbose logs: $OUTPUT_ROOT/awesome_verbose_*/verbose_*.log"
echo "Input: $INPUT_TSV"
echo "Output root: $OUTPUT_ROOT"
