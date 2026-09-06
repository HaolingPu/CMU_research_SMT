#!/usr/bin/env bash
#SBATCH --job-name=resample_manifest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --time=00:30:00

# Reuse an existing converted manifest and its audio clips while creating a
# count-matched training manifest for a controlled comparison.
set -euo pipefail

SOURCE_MANIFEST="${SOURCE_MANIFEST:?SOURCE_MANIFEST required}"
TARGET_MANIFEST="${TARGET_MANIFEST:?TARGET_MANIFEST required}"
SAMPLE_N="${SAMPLE_N:-12500}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate evaluation
export PYTHONNOUSERSITE=1

mkdir -p "$(dirname "${TARGET_MANIFEST}")"

python /home/haolingp/CMU_research_SMT/scripts/train/sample_jsonl.py \
  "${SAMPLE_N}" "${SOURCE_MANIFEST}" "${TARGET_MANIFEST}" \
  --seed "${SAMPLE_SEED}"

ACTUAL=$(wc -l < "${TARGET_MANIFEST}")
if [[ "${ACTUAL}" -ne "${SAMPLE_N}" ]]; then
  echo "ERROR: expected ${SAMPLE_N} rows, found ${ACTUAL}" >&2
  exit 1
fi

echo "Verified ${ACTUAL} rows in ${TARGET_MANIFEST} (seed=${SAMPLE_SEED})"
