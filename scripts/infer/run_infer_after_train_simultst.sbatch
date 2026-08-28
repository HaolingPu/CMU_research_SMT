#!/usr/bin/env bash
#SBATCH --job-name=infer_launch_tst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:L40S:1
#SBATCH --qos=normal
#SBATCH --partition=general
#SBATCH --time=00:20:00
#SBATCH -o /home/haolingp/CMU_research_SMT/scripts/infer/slurm_logs/infer_launch_tst_%j.out
#SBATCH -e /home/haolingp/CMU_research_SMT/scripts/infer/slurm_logs/infer_launch_tst_%j.err

# Simul-tst-COMMON twin of run_infer_after_train_generic.sbatch. Fires
# afterok:train, finds the HF export of ${EXP}, launches the 4-seg simultst
# inference array, registers it in ckpts_simultst.txt, chains simultst eval.
set -e
cd /home/haolingp/CMU_research_SMT/scripts/infer

EXP="${EXP:?EXP required}"
BASE=/data/user_data/haolingp/ckpts/infinisst-omni/${EXP}
HF=$(ls -td "${BASE}"/v*-hf 2>/dev/null | head -1)
if [[ -z "${HF}" ]]; then echo "ERROR: no -hf export under ${BASE}"; ls -la "${BASE}" || true; exit 1; fi
echo "HF export: ${HF}"

INFER_JID=$(sbatch --parsable infer_slurm_simultst.sh "${HF}/" Standard)
echo "simultst infer array: ${INFER_JID}"

REL="${EXP}/$(basename "${HF}")/"
if ! grep -qF "${REL}" ckpts_simultst.txt; then echo "${REL}" >> ckpts_simultst.txt; echo "appended ${REL} to ckpts_simultst.txt"; fi

EVAL_JID=$(sbatch --parsable --dependency=afterok:${INFER_JID} run_eval_all_ckpts_simultst.sbatch)
echo "simultst eval: ${EVAL_JID} (afterok:${INFER_JID})"
