#!/usr/bin/env bash
#SBATCH --job-name=infer_launch_salami_ja
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=4G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=00:15:00
#SBATCH --exclude=babel-t9-16
#SBATCH -o /home/haolingp/CMU_research_SMT/scripts/train/slurm_logs/infer_launch_salami_ja_%j.out
#SBATCH -e /home/haolingp/CMU_research_SMT/scripts/train/slurm_logs/infer_launch_salami_ja_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

# en->ja launcher: discover the latest HF checkpoint for ${EXP_NAME}, append it
# to ckpts_ja.txt, then chain:
#   1) infer_slurm_ja.sh     -> simuleval (4-array, en-ja, seg 960/1920/2880/3840)
#   2) run_eval_all_ckpts_ja.sbatch (afterok)  -> omnisteval longform (BLEU/COMET)
#
# Required env vars:
#   EXP_NAME       -- e.g. gigaspeech-ja-EAST-latency2mult-s_origin-bsz4_ja
#   PROMPT_TYPE    -- Standard or EAST (default Standard)

set -e

EXP_NAME="${EXP_NAME:?EXP_NAME required}"
PROMPT_TYPE="${PROMPT_TYPE:-Standard}"

CKPT_ROOT=/data/user_data/haolingp/ckpts/infinisst-omni
BASE_DIR=${CKPT_ROOT}/${EXP_NAME}
LATEST_HF=$(ls -td "${BASE_DIR}"/v*-hf 2>/dev/null | head -n 1)

if [[ -z "${LATEST_HF}" ]] || [[ ! -d "${LATEST_HF}" ]]; then
  echo "ERROR: no HF ckpt under ${BASE_DIR}/v*-hf"
  ls -la "${BASE_DIR}" 2>/dev/null
  exit 1
fi

echo "===== INFER LAUNCH (ja) START $(date) ====="
echo "EXP_NAME    : ${EXP_NAME}"
echo "MODEL_PATH  : ${LATEST_HF}"
echo "PROMPT_TYPE : ${PROMPT_TYPE}"

CKPTS_TXT=/home/haolingp/CMU_research_SMT/scripts/infer/ckpts_ja.txt
CKPT_REL="${EXP_NAME}/$(basename "${LATEST_HF}")/"
if ! grep -qxF "${CKPT_REL}" "${CKPTS_TXT}" 2>/dev/null; then
  echo "${CKPT_REL}" >> "${CKPTS_TXT}"
  echo "[ckpts_ja.txt] appended ${CKPT_REL}"
else
  echo "[ckpts_ja.txt] ${CKPT_REL} already present"
fi

cd /home/haolingp/CMU_research_SMT/scripts/infer

# Stage A: simuleval (chunk-level), ja
INFER_JOB=$(sbatch --parsable infer_slurm_ja.sh "${LATEST_HF}/" "${PROMPT_TYPE}")
echo "[infer array (ja)] ${INFER_JOB}"

# Stage B: longform eval AFTER simuleval done
EVAL_JOB=$(sbatch --parsable \
  --dependency="afterok:${INFER_JOB}" \
  /home/haolingp/CMU_research_SMT/scripts/infer/run_eval_all_ckpts_ja.sbatch)
echo "[longform eval (ja)] ${EVAL_JOB}  (afterok:${INFER_JOB})"

echo "===== INFER LAUNCH (ja) DONE $(date) ====="
echo "After completion, results are in:"
echo "  simuleval (chunk-level): ${LATEST_HF}/evaluation/acl_6060/en-ja/seg<N>/scores.tsv"
echo "  longform (BLEU/COMET)  : ${LATEST_HF}/evaluation/acl_6060/en-ja/seg<N>/segmentation_output/scores.tsv"
