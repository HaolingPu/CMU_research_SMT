#!/usr/bin/env bash
#SBATCH --job-name=teacher_prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --requeue
#SBATCH --time=1-00:00:00
#SBATCH -o /home/haolingp/CMU_research_SMT/scripts/synth/slurm_logs/teacher_prepare_%j.out
#SBATCH -e /home/haolingp/CMU_research_SMT/scripts/synth/slurm_logs/teacher_prepare_%j.err

set -euo pipefail

METHOD="${METHOD:?METHOD must be east or simul-must-c}"
BASE="${BASE:?BASE is required}"
CODE_SNAPSHOT="${CODE_SNAPSHOT:?CODE_SNAPSHOT is required}"
INPUT_TSV="${INPUT_TSV:-/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv}"
MFA_DIR="${MFA_DIR:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/gigaspeech_mfa_textgrid}"
MFA_CORPUS="${MFA_CORPUS:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/gigaspeech_mfa_corpus}"
NUM_QE_SHARDS="${NUM_QE_SHARDS:-24}"

case "${BASE}" in
  */qwen36_teacher_ablation/*) ;;
  *) echo "Refusing unsafe BASE outside qwen36_teacher_ablation: ${BASE}" >&2; exit 2 ;;
esac

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate SMT

RAW="${BASE}/llm_output_raw"
RAW_FIXED="${BASE}/llm_output_raw_fixed"
rm -rf "${RAW_FIXED}"

if [[ "${METHOD}" == "east" ]]; then
  GOOD_FIXED="${BASE}/good_train_xl_east_fixed.jsonl"
  GOOD_MFA="${BASE}/good_train_xl_east_mfa.jsonl"
  BAD_MFA="${BASE}/bad_train_xl_east_mfa.jsonl"
  MERGED="${BASE}/llm_output_merged_fixed"
  STREAM="${BASE}/streaming_EAST_dataset"

  python "${CODE_SNAPSHOT}/fix_llm_raw.py" \
    --in_dir "${RAW}" \
    --out_dir "${RAW_FIXED}" \
    --out_good_jsonl "${GOOD_FIXED}" \
    --sync_zh_punct \
    --zh_punct_allow_insert

  rm -rf "${MERGED}"
  python "${CODE_SNAPSHOT}/post_process_llm_output_gigaspeech.py" \
    --input-dir "${RAW_FIXED}" \
    --output-dir "${MERGED}" \
    --overwrite

  python "${CODE_SNAPSHOT}/find_bad_json_gigaspeech.py" \
    --llm-dir "${MERGED}" \
    --mfa-dir "${MFA_DIR}" \
    --corpus-dir "${MFA_CORPUS}" \
    --good-jsonl "${GOOD_MFA}" \
    --bad-jsonl "${BAD_MFA}"
elif [[ "${METHOD}" == "simul-must-c" ]]; then
  GOOD_FIXED="${BASE}/good_train_xl_salami_fixed.jsonl"
  GOOD_MFA="${BASE}/good_train_xl_salami_mfa.jsonl"
  BAD_MFA="${BASE}/bad_train_xl_salami_mfa.jsonl"
  OFFLINE="${BASE}/llm_output_offline_fixed"
  MERGED="${BASE}/llm_output_offline_merged_fixed"
  STREAM="${BASE}/streaming_salami_dataset"

  python "${CODE_SNAPSHOT}/fix_llm_raw.py" \
    --in_dir "${RAW}" \
    --out_dir "${RAW_FIXED}" \
    --out_good_jsonl "${GOOD_FIXED}" \
    --sync_zh_punct \
    --zh_punct_allow_insert \
    --filter_zh_punct \
    --zh_excess_threshold 2

  rm -rf "${OFFLINE}" "${MERGED}"
  python "${CODE_SNAPSHOT}/salami/map_salami_to_offline_gigaspeech.py" \
    --input_dir "${RAW_FIXED}" \
    --output_dir "${OFFLINE}"
  python "${CODE_SNAPSHOT}/post_process_llm_output_gigaspeech.py" \
    --input-dir "${OFFLINE}" \
    --output-dir "${MERGED}" \
    --overwrite
  python "${CODE_SNAPSHOT}/find_bad_json_gigaspeech.py" \
    --llm-dir "${MERGED}" \
    --mfa-dir "${MFA_DIR}" \
    --corpus-dir "${MFA_CORPUS}" \
    --good-jsonl "${GOOD_MFA}" \
    --bad-jsonl "${BAD_MFA}" \
    --allow-one-word
else
  echo "Unknown METHOD=${METHOD}" >&2
  exit 2
fi

rm -rf "${STREAM}"
python "${CODE_SNAPSHOT}/multi_trajectory_gigaspeech.py" \
  --llm-dir "${MERGED}" \
  --mfa-dir "${MFA_DIR}" \
  --good-jsonl "${GOOD_MFA}" \
  --output-dir "${STREAM}" \
  --chunk-ms 960 \
  --overwrite

python "${CODE_SNAPSHOT}/check_streaming_dataset.py" \
  --stream_dir "${STREAM}" \
  --tsv "${INPUT_TSV}" \
  --report "${BASE}/check_streaming_report.json" \
  --zh_excess_threshold 2 || true

conda deactivate
conda activate metricx
python "${CODE_SNAPSHOT}/convert_metricx_gigaspeech.py" \
  --stream_dir "${STREAM}" \
  --output "${BASE}/metricx_input.jsonl"

rm -rf "${BASE}/metricx_shards"
mkdir -p "${BASE}/metricx_shards"
split -d -a 2 -n "l/${NUM_QE_SHARDS}" \
  "${BASE}/metricx_input.jsonl" \
  "${BASE}/metricx_shards/input_"

SHARD_COUNT=$(find "${BASE}/metricx_shards" -maxdepth 1 -type f -name 'input_*' | wc -l)
if [[ "${SHARD_COUNT}" -ne "${NUM_QE_SHARDS}" ]]; then
  echo "Expected ${NUM_QE_SHARDS} MetricX shards, found ${SHARD_COUNT}" >&2
  exit 1
fi
echo "Prepared ${SHARD_COUNT} MetricX shards at $(date --iso-8601=seconds)"
