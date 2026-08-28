#!/usr/bin/env bash
#SBATCH --job-name=conv2swift_LA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=24G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --exclude=babel-t9-16,babel-p5-24
#SBATCH -o /home/haolingp/CMU_research_SMT/scripts/train/slurm_logs/conv2swift_LA_%j.out
#SBATCH -e /home/haolingp/CMU_research_SMT/scripts/train/slurm_logs/conv2swift_LA_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

# Build LA training manifest + chunked audio from a directory of post-filter
# local-agreement JSONs, then random-downsample to SAMPLE_N rows so the run
# is comparable in size to the consensus / PA baselines.
#
# Required env vars:
#   MANIFEST_ROOT  -- dir of <utt_id>.json files (post QE+LR filter)
#   VARIANT_TAG    -- e.g. 40k
# Optional env vars:
#   SAMPLE_N       -- random downsample target (default 12500). Set to 0 to skip.
#   SAMPLE_SEED    -- seed for downsampling (default 42).

set -e

MANIFEST_ROOT="${MANIFEST_ROOT:?MANIFEST_ROOT required}"
VARIANT_TAG="${VARIANT_TAG:?VARIANT_TAG required}"
SAMPLE_N="${SAMPLE_N:-12500}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

mkdir -p /home/haolingp/CMU_research_SMT/scripts/train/slurm_logs

# Fast-fail if the home NFS mount is missing on this node.
if [ ! -f /home/haolingp/miniconda3/etc/profile.d/conda.sh ]; then
    echo "[FATAL] /home/haolingp/miniconda3 not visible on $(hostname) — likely NFS not mounted. Aborting." >&2
    exit 1
fi

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate evaluation
export PYTHONNOUSERSITE=1

echo "===== CONVERT2SWIFT_LA START $(date) ====="
echo "MANIFEST_ROOT : ${MANIFEST_ROOT}"
echo "VARIANT_TAG   : ${VARIANT_TAG}"
echo "SAMPLE_N      : ${SAMPLE_N}"
echo "SAMPLE_SEED   : ${SAMPLE_SEED}"
echo "json count    : $(ls ${MANIFEST_ROOT}/*.json 2>/dev/null | wc -l)"

python /home/haolingp/CMU_research_SMT/scripts/train/convert2swift_LA.py \
  --manifest-root "${MANIFEST_ROOT}" \
  --variant-tag   "${VARIANT_TAG}"

DEST=/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests
OUT_MANIFEST=${DEST}/train_s_zh-LA-${VARIANT_TAG}.jsonl
FULL_MANIFEST=${DEST}/train_s_zh-LA-${VARIANT_TAG}_full.jsonl

FULL_LINES=$(wc -l < "${OUT_MANIFEST}")
echo "full manifest lines  : ${FULL_LINES}"

if [[ "${SAMPLE_N}" == "0" ]]; then
  echo "SAMPLE_N=0 -> skipping random downsample"
elif [[ "${FULL_LINES}" -le "${SAMPLE_N}" ]]; then
  echo "full ${FULL_LINES} <= SAMPLE_N=${SAMPLE_N}, no downsample needed; copying to *_full.jsonl for record"
  cp "${OUT_MANIFEST}" "${FULL_MANIFEST}"
else
  echo "downsampling ${FULL_LINES} -> ${SAMPLE_N} (seed=${SAMPLE_SEED})"
  cp "${OUT_MANIFEST}" "${FULL_MANIFEST}"
  python /home/haolingp/CMU_research_SMT/scripts/train/sample_jsonl.py \
    "${SAMPLE_N}" "${FULL_MANIFEST}" "${OUT_MANIFEST}" --seed "${SAMPLE_SEED}"
fi

echo
echo "[FINAL]"
echo "  training manifest : ${OUT_MANIFEST}  ($(wc -l < ${OUT_MANIFEST}) lines)"
echo "  full backup       : ${FULL_MANIFEST}  ($(wc -l < ${FULL_MANIFEST} 2>/dev/null || echo 0) lines)"
echo "  audio clips       : ${DEST%/manifests}/audio_clips_zh_LA-${VARIANT_TAG}/"
echo "===== CONVERT2SWIFT_LA DONE $(date) ====="
