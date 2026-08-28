#!/usr/bin/env bash
#SBATCH --job-name=win3_infer_launch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:L40S:1
#SBATCH --qos=normal
#SBATCH --partition=general
#SBATCH --time=00:20:00
#SBATCH -o /home/haolingp/CMU_research_SMT/scripts/infer/slurm_logs/win3_infer_launch_%j.out
#SBATCH -e /home/haolingp/CMU_research_SMT/scripts/infer/slurm_logs/win3_infer_launch_%j.err

# Fires afterok:train. Finds the timestamped HF export of the win3 run, launches
# the 4-seg inference array, registers it in ckpts.txt, then chains eval afterok:infer.
set -e
cd /home/haolingp/CMU_research_SMT/scripts/infer

EXP=gigaspeech-zh-consensus-INTERSECT-win3-fixed-s-bsz4
BASE=/data/user_data/haolingp/ckpts/infinisst-omni/${EXP}
HF=$(ls -td "${BASE}"/v*-hf 2>/dev/null | head -1)
if [[ -z "${HF}" ]]; then echo "ERROR: no -hf export under ${BASE}"; ls -la "${BASE}" || true; exit 1; fi
echo "win3 HF export: ${HF}"

INFER_JID=$(sbatch --parsable infer_slurm.sh "${HF}/" Standard)
echo "infer array: ${INFER_JID}"

REL="${EXP}/$(basename "${HF}")/"
if ! grep -qF "${REL}" ckpts.txt; then echo "${REL}" >> ckpts.txt; echo "appended ${REL} to ckpts.txt"; fi

EVAL_JID=$(sbatch --parsable --dependency=afterok:${INFER_JID} run_eval_all_ckpts.sbatch)
echo "eval: ${EVAL_JID} (afterok:${INFER_JID})"
echo "win3 infer=${INFER_JID} eval=${EVAL_JID}" > /tmp/win3_infer_eval_jids.txt
