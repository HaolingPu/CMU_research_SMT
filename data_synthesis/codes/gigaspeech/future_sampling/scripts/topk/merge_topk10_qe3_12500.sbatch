#!/usr/bin/env bash
#SBATCH --job-name=topk10_merge12500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=8G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/topk10_merge12500_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/topk10_merge12500_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

MAIN_DIR="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/topk/consensus_decoding_en_zh_top_10/consensus_decoding_en_zh_top10_qe3"
TOPUP_DIR="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/topk/consensus_decoding_en_zh_top_10/job_7064029-qe3"
OUT_DIR="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/topk/consensus_decoding_en_zh_top_10/consensus_decoding_en_zh_top10_qe3_12500"
TARGET_COUNT=12500

mkdir -p "${OUT_DIR}"

echo "main=${MAIN_DIR}"
echo "topup=${TOPUP_DIR}"
echo "out=${OUT_DIR}"
echo "target=${TARGET_COUNT}"

if [[ ! -d "${MAIN_DIR}" ]]; then
  echo "ERROR: missing main dir: ${MAIN_DIR}"
  exit 1
fi

if [[ ! -d "${TOPUP_DIR}" ]]; then
  echo "ERROR: missing topup dir: ${TOPUP_DIR}"
  exit 1
fi

find "${MAIN_DIR}" -maxdepth 1 -type f -name '*.json' -print | sort | while read -r path; do
  cp -n "${path}" "${OUT_DIR}/"
done

count=$(find "${OUT_DIR}" -maxdepth 1 -type f -name '*.json' | wc -l)
echo "after main copy: ${count}"

find "${TOPUP_DIR}" -maxdepth 1 -type f -name '*.json' -print | sort | while read -r path; do
  count=$(find "${OUT_DIR}" -maxdepth 1 -type f -name '*.json' | wc -l)
  if [[ "${count}" -ge "${TARGET_COUNT}" ]]; then
    break
  fi
  cp -n "${path}" "${OUT_DIR}/"
done

final_count=$(find "${OUT_DIR}" -maxdepth 1 -type f -name '*.json' | wc -l)
echo "final_count=${final_count}"

if [[ "${final_count}" -lt "${TARGET_COUNT}" ]]; then
  echo "ERROR: not enough QE<=3 files to reach ${TARGET_COUNT}; got ${final_count}"
  exit 1
fi

echo "done"
