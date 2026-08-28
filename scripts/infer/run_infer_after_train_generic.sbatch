#!/usr/bin/env bash
#SBATCH --job-name=infer_launch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:L40S:1
#SBATCH --qos=normal
#SBATCH --partition=general
#SBATCH --time=00:20:00
#SBATCH -o /home/haolingp/CMU_research_SMT/scripts/infer/slurm_logs/infer_launch_%j.out
#SBATCH -e /home/haolingp/CMU_research_SMT/scripts/infer/slurm_logs/infer_launch_%j.err

# Fires afterok:train. Finds the timestamped HF export of ${EXP}, launches the
# 4-seg inference array, registers it in ckpts.txt, then chains eval afterok:infer.
set -e
cd /home/haolingp/CMU_research_SMT/scripts/infer

EXP="${EXP:?EXP required}"
CHILD_PARTITION="${CHILD_PARTITION:-general}"
CHILD_GPU_QOS="${CHILD_GPU_QOS:-normal}"
CKPTS_FILE="${CKPTS_FILE:-/home/haolingp/slurm_runs/${EXP}/ckpts.txt}"
BASE=/data/user_data/haolingp/ckpts/infinisst-omni/${EXP}
HF=$(ls -td "${BASE}"/v*-hf 2>/dev/null | head -1)
if [[ -z "${HF}" ]]; then echo "ERROR: no -hf export under ${BASE}"; ls -la "${BASE}" || true; exit 1; fi
echo "HF export: ${HF}"

mkdir -p "$(dirname "${CKPTS_FILE}")"

INFER_JID=$(sbatch --parsable \
  --partition="${CHILD_PARTITION}" \
  --qos="${CHILD_GPU_QOS}" \
  --requeue \
  infer_slurm.sh "${HF}/" Standard)
echo "infer array: ${INFER_JID}"

REL="${EXP}/$(basename "${HF}")/"
if ! grep -qF "${REL}" "${CKPTS_FILE}" 2>/dev/null; then
  echo "${REL}" >> "${CKPTS_FILE}"
  echo "appended ${REL} to ${CKPTS_FILE}"
fi

EVAL_JID=$(sbatch --parsable \
  --dependency=afterok:${INFER_JID} \
  --partition="${CHILD_PARTITION}" \
  --qos="${CHILD_GPU_QOS}" \
  --requeue \
  --export="ALL,CKPTS_FILE=${CKPTS_FILE}" \
  run_eval_all_ckpts.sbatch)
echo "eval: ${EVAL_JID} (afterok:${INFER_JID})"
echo "${EXP} infer=${INFER_JID} eval=${EVAL_JID}" > /tmp/${EXP}_infer_eval_jids.txt
