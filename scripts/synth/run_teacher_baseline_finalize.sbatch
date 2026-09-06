#!/usr/bin/env bash
#SBATCH --job-name=teacher_finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH -o /home/haolingp/CMU_research_SMT/scripts/synth/slurm_logs/teacher_finalize_%j.out
#SBATCH -e /home/haolingp/CMU_research_SMT/scripts/synth/slurm_logs/teacher_finalize_%j.err

set -euo pipefail

METHOD="${METHOD:?METHOD must be east or simul-must-c}"
BASE="${BASE:?BASE is required}"
CODE_SNAPSHOT="${CODE_SNAPSHOT:?CODE_SNAPSHOT is required}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:?OUTPUT_MANIFEST is required}"
POOL_MANIFEST="${POOL_MANIFEST:?POOL_MANIFEST is required}"
TEMPLATE_MANIFEST="${TEMPLATE_MANIFEST:?TEMPLATE_MANIFEST is required}"
NUM_QE_SHARDS="${NUM_QE_SHARDS:-24}"
SAMPLE_SIZE="${SAMPLE_SIZE:-12500}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
INPUT_TSV="${INPUT_TSV:-/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv}"
REPO=/home/haolingp/CMU_research_SMT

case "${BASE}" in
  */qwen36_teacher_ablation/*) ;;
  *) echo "Refusing unsafe BASE outside qwen36_teacher_ablation: ${BASE}" >&2; exit 2 ;;
esac

for index in $(seq 0 $((NUM_QE_SHARDS - 1))); do
  shard=$(printf '%02d' "${index}")
  test -s "${BASE}/metricx_shards/output_${shard}.jsonl"
done

cat "${BASE}"/metricx_shards/output_*.jsonl > "${BASE}/metricx_output.jsonl"

if [[ "${METHOD}" == "east" ]]; then
  THRESHOLD=3.0
  FILTERED="${BASE}/metricx_filtered_t3.0.jsonl"
  STREAM="${BASE}/streaming_EAST_dataset"
  FINAL="${BASE}/final_jsonl_east"
  BUILD_METHOD=east-even
elif [[ "${METHOD}" == "simul-must-c" ]]; then
  THRESHOLD=5.0
  FILTERED="${BASE}/metricx_filtered_t5.0.jsonl"
  STREAM="${BASE}/streaming_salami_dataset"
  FINAL="${BASE}/final_jsonl_salami"
  BUILD_METHOD=simul-must-c
else
  echo "Unknown METHOD=${METHOD}" >&2
  exit 2
fi

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate metricx
python "${CODE_SNAPSHOT}/filter_metricx_gigaspeech.py" \
  --input "${BASE}/metricx_output.jsonl" \
  --output "${FILTERED}" \
  --threshold "${THRESHOLD}"

conda deactivate
conda activate SMT
rm -rf "${FINAL}"
python "${CODE_SNAPSHOT}/final_output_gigaspeech.py" \
  --metricx_jsonl "${FILTERED}" \
  --stream_dir "${STREAM}" \
  --output_dir "${FINAL}"

python "${CODE_SNAPSHOT}/check_salami_final.py" \
  --tsv "${INPUT_TSV}" \
  --manifest "${FILTERED}" \
  --final_dir "${FINAL}" \
  --report "${BASE}/check_final_report.json" || true

python "${REPO}/scripts/train/build_teacher_ablation_manifest.py" \
  --method "${BUILD_METHOD}" \
  --final-root "${FINAL}" \
  --pool-manifest "${POOL_MANIFEST}" \
  --template-manifest "${TEMPLATE_MANIFEST}" \
  --output "${OUTPUT_MANIFEST}" \
  --report "${BASE}/manifest_build_report.json" \
  --sample-size "${SAMPLE_SIZE}" \
  --seed "${SAMPLE_SEED}"

COUNT=$(grep -cve '^[[:space:]]*$' "${OUTPUT_MANIFEST}")
if [[ "${COUNT}" -ne "${SAMPLE_SIZE}" ]]; then
  echo "Training manifest has ${COUNT} rows, expected ${SAMPLE_SIZE}" >&2
  exit 1
fi
echo "Verified frozen training size: ${COUNT}"
