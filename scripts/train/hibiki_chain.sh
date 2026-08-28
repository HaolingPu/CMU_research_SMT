#!/usr/bin/env bash
#SBATCH --job-name=hibiki_chain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:L40S:1
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=00:10:00
#SBATCH -o /home/haolingp/CMU_research_SMT/scripts/train/slurm_logs/%j.out
#SBATCH -e /home/haolingp/CMU_research_SMT/scripts/train/slurm_logs/%j.err

# Fires after hibiki training (afterok). Discovers the exported -hf ckpt,
# submits streaming inference (Standard prompt, 4 seg sizes), registers it in
# ckpts.txt, then chains eval (afterok on the infer array job).
set -e

CKPT_ROOT=/data/user_data/haolingp/ckpts/infinisst-omni
EXP=gigaspeech-zh-hibiki-s-bsz4
INFER_DIR=/home/haolingp/CMU_research_SMT/scripts/infer

HF=$(ls -td "${CKPT_ROOT}/${EXP}"/v*-hf 2>/dev/null | head -n 1)
if [ -z "$HF" ]; then
    echo "ERROR: no -hf export found for ${EXP}; training may have failed." >&2
    exit 1
fi
echo "hibiki -hf: $HF"

cd "$INFER_DIR"
INFER_JOB=$(sbatch --parsable infer_slurm.sh "$HF" Standard)
echo "submitted infer array job: $INFER_JOB"

# register in ckpts.txt (relative to CKPT_ROOT), idempotent
REL="${HF#${CKPT_ROOT}/}/"
if ! grep -qF "$REL" "${INFER_DIR}/ckpts.txt"; then
    printf '%s\n' "$REL" >> "${INFER_DIR}/ckpts.txt"
    echo "added to ckpts.txt: $REL"
else
    echo "already in ckpts.txt: $REL"
fi

EVAL_JOB=$(sbatch --parsable --dependency=afterok:${INFER_JOB} run_eval_all_ckpts.sbatch)
echo "submitted eval job: $EVAL_JOB (afterok:${INFER_JOB})"
