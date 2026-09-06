#!/usr/bin/env bash
#SBATCH --job-name=teacher_validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --requeue
#SBATCH --time=02:00:00
#SBATCH -o /home/haolingp/CMU_research_SMT/scripts/synth/slurm_logs/teacher_validate_%j.out
#SBATCH -e /home/haolingp/CMU_research_SMT/scripts/synth/slurm_logs/teacher_validate_%j.err

set -euo pipefail

METHOD="${METHOD:?METHOD must be east or simul-must-c}"
BASE="${BASE:?BASE is required}"
INPUT_TSV="${INPUT_TSV:-/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv}"
VALIDATION_PYTHON="${VALIDATION_PYTHON:-/data/user_data/haolingp/conda_envs/gemma4/bin/python}"
REPO=/home/haolingp/CMU_research_SMT

case "${BASE}" in
  */qwen36_teacher_ablation/*) ;;
  *) echo "Refusing unsafe BASE outside qwen36_teacher_ablation: ${BASE}" >&2; exit 2 ;;
esac

EXPECTED_COUNT=$(($(wc -l < "${INPUT_TSV}") - 1))
if [[ "${EXPECTED_COUNT}" -le 0 ]]; then
  echo "Input TSV has no data rows: ${INPUT_TSV}" >&2
  exit 2
fi

ARGS=(
  --method "${METHOD}"
  --output-root "${BASE}/llm_output_raw"
  --expected-count "${EXPECTED_COUNT}"
)
if [[ "${ALLOW_RECORDED_ERRORS:-1}" == "1" ]]; then
  ARGS+=(--allow-recorded-errors)
fi

"${VALIDATION_PYTHON}" \
  "${REPO}/scripts/synth/validate_teacher_raw_output.py" \
  "${ARGS[@]}"
