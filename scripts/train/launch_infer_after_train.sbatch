#!/usr/bin/env bash
#SBATCH --job-name=infer_launch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=4G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=00:15:00
#SBATCH --exclude=babel-t9-16
#SBATCH -o /home/haolingp/CMU_research_SMT/scripts/train/slurm_logs/infer_launch_%j.out
#SBATCH -e /home/haolingp/CMU_research_SMT/scripts/train/slurm_logs/infer_launch_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

# Discover latest HF checkpoint for ${EXP_NAME}, append it to ckpts.txt, then chain:
#   1) infer_slurm.sh  -> simuleval on 4 segment sizes (chunk-level BLEU/AL/...)
#   2) run_eval_all_ckpts.sbatch (afterok)  -> omnisteval longform (BLEU + COMET + streamLAAL)
#
# Required env vars:
#   EXP_NAME       -- e.g. gigaspeech-zh-consensus-topk5_k4-s-bsz4
#   PROMPT_TYPE    -- Standard or EAST (default Standard)

set -e

EXP_NAME="${EXP_NAME:?EXP_NAME required}"
PROMPT_TYPE="${PROMPT_TYPE:-Standard}"

BASE_DIR=/data/user_data/haolingp/ckpts/infinisst-omni/${EXP_NAME}
LATEST_HF=$(ls -td "${BASE_DIR}"/v*-hf 2>/dev/null | head -n 1)

if [[ -z "${LATEST_HF}" ]] || [[ ! -d "${LATEST_HF}" ]]; then
  echo "ERROR: no HF ckpt under ${BASE_DIR}/v*-hf"
  ls -la "${BASE_DIR}" 2>/dev/null
  exit 1
fi

echo "===== INFER LAUNCH START $(date) ====="
echo "EXP_NAME    : ${EXP_NAME}"
echo "MODEL_PATH  : ${LATEST_HF}"
echo "PROMPT_TYPE : ${PROMPT_TYPE}"

# Add this ckpt (relative path from CKPT_ROOT) to ckpts.txt so longform eval picks it up
CKPTS_TXT=/home/haolingp/CMU_research_SMT/scripts/infer/ckpts.txt
CKPT_REL="${EXP_NAME}/$(basename "${LATEST_HF}")/"
if ! grep -qxF "${CKPT_REL}" "${CKPTS_TXT}" 2>/dev/null; then
  echo "${CKPT_REL}" >> "${CKPTS_TXT}"
  echo "[ckpts.txt] appended ${CKPT_REL}"
else
  echo "[ckpts.txt] ${CKPT_REL} already present"
fi

cd /home/haolingp/CMU_research_SMT/scripts/infer

# Stage A: simuleval (chunk-level)
INFER_JOB=$(sbatch --parsable infer_slurm.sh "${LATEST_HF}/" "${PROMPT_TYPE}")
echo "[infer array]      ${INFER_JOB}"

# Stage B: longform eval (omnisteval -> streamLAAL + COMET) AFTER simuleval done
EVAL_JOB=$(sbatch --parsable \
  --dependency="afterok:${INFER_JOB}" \
  /home/haolingp/CMU_research_SMT/scripts/infer/run_eval_all_ckpts.sbatch)
echo "[longform eval]    ${EVAL_JOB}  (afterok:${INFER_JOB})"

echo "===== INFER LAUNCH DONE $(date) ====="
echo "After completion, results are in:"
echo "  simuleval (chunk-level): ${LATEST_HF}/evaluation/acl_6060/en-zh/seg<N>/scores.tsv"
echo "  longform (streamLAAL)  : ${LATEST_HF}/evaluation/acl_6060/en-zh/seg<N>/segmentation_output/scores.tsv"
