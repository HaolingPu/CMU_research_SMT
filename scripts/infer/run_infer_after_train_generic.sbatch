#!/usr/bin/env bash
#SBATCH --job-name=infer_launch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --qos=cpu_qos
#SBATCH --partition=cpu
#SBATCH --time=00:20:00
#SBATCH -o /home/haolingp/CMU_research_SMT/scripts/infer/slurm_logs/infer_launch_%j.out
#SBATCH -e /home/haolingp/CMU_research_SMT/scripts/infer/slurm_logs/infer_launch_%j.err

# Fires afterok:train. Finds the timestamped HF export of ${EXP}, launches the
# 4-seg inference array, registers it in ckpts.txt, then chains eval afterok:infer.
# With RUN_SIMULTST=1 it also launches the Simul-tst-COMMON (monotonic
# references) inference array and its eval as an independent second chain.
set -e
cd /home/haolingp/CMU_research_SMT/scripts/infer

EXP="${EXP:?EXP required}"
CHILD_PARTITION="${CHILD_PARTITION:-general}"
CHILD_GPU_QOS="${CHILD_GPU_QOS:-normal}"
CKPTS_FILE="${CKPTS_FILE:-/home/haolingp/slurm_runs/${EXP}/ckpts.txt}"
RUN_SIMULTST="${RUN_SIMULTST:-0}"
CKPTS_SIMULTST_FILE="${CKPTS_SIMULTST_FILE:-/home/haolingp/slurm_runs/${EXP}/ckpts_simultst.txt}"
PIPELINE_MANIFEST="${PIPELINE_MANIFEST:-}"
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
if [[ -n "${PIPELINE_MANIFEST}" ]]; then
  {
    echo "infer=${INFER_JID}"
    echo "comet_eval=${EVAL_JID}"
    echo "comet_model=Unbabel/XCOMET-XL"
  } >> "${PIPELINE_MANIFEST}"
fi

if [[ "${RUN_SIMULTST}" == "1" ]]; then
  # Second chain: Simul-tst-COMMON inference (4 segs x 2 GPUs) -> simultst eval.
  # Runs concurrently with the ACL 6060 chain; both read the same HF export.
  TST_INFER_JID=$(sbatch --parsable \
    --partition="${CHILD_PARTITION}" \
    --qos="${CHILD_GPU_QOS}" \
    --requeue \
    infer_slurm_simultst.sh "${HF}/" Standard)
  echo "simultst infer array: ${TST_INFER_JID}"

  if ! grep -qF "${REL}" "${CKPTS_SIMULTST_FILE}" 2>/dev/null; then
    echo "${REL}" >> "${CKPTS_SIMULTST_FILE}"
    echo "appended ${REL} to ${CKPTS_SIMULTST_FILE}"
  fi

  TST_EVAL_JID=$(sbatch --parsable \
    --dependency=afterok:${TST_INFER_JID} \
    --partition="${CHILD_PARTITION}" \
    --qos="${CHILD_GPU_QOS}" \
    --requeue \
    --export="ALL,CKPTS_FILE=${CKPTS_SIMULTST_FILE}" \
    run_eval_all_ckpts_simultst.sbatch)
  echo "simultst eval: ${TST_EVAL_JID} (afterok:${TST_INFER_JID})"
  echo "${EXP} simultst_infer=${TST_INFER_JID} simultst_eval=${TST_EVAL_JID}" >> /tmp/${EXP}_infer_eval_jids.txt
  if [[ -n "${PIPELINE_MANIFEST}" ]]; then
    {
      echo "simultst_infer=${TST_INFER_JID}"
      echo "simultst_eval=${TST_EVAL_JID}"
      echo "simultst_ckpts_file=${CKPTS_SIMULTST_FILE}"
      echo "simultst_refs=diagnostic_grade_2026-07-11_rebuild"
    } >> "${PIPELINE_MANIFEST}"
  fi
fi
