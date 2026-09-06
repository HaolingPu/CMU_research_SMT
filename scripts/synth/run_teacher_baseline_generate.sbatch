#!/usr/bin/env bash
#SBATCH --job-name=teacher_gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=180G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --time=2-00:00:00
#SBATCH --array=0-0
#SBATCH -o /home/haolingp/CMU_research_SMT/scripts/synth/slurm_logs/teacher_gen_%A_%a.out
#SBATCH -e /home/haolingp/CMU_research_SMT/scripts/synth/slurm_logs/teacher_gen_%A_%a.err

set -euo pipefail

METHOD="${METHOD:?METHOD must be east or simul-must-c}"
BASE="${BASE:?BASE is required}"
CODE_SNAPSHOT="${CODE_SNAPSHOT:?CODE_SNAPSHOT is required}"
MODEL_PATH="${MODEL_PATH:-/data/user_data/haolingp/models/Qwen3.6-35B-A3B-FP8}"
GENERATION_ENV="${GENERATION_ENV:-/data/user_data/haolingp/conda_envs/gemma4}"
LEGACY_RUNNER="/home/haolingp/CMU_research_SMT/scripts/synth/run_legacy_vllm_script.py"
INPUT_TSV="${INPUT_TSV:-/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv}"
NUM_TASKS="${NUM_TASKS:-24}"
TP="${TP:-1}"
MAX_ROWS="${MAX_ROWS:-}"
OUT_ROOT="${BASE}/llm_output_raw"

case "${BASE}" in
  */qwen36_teacher_ablation/*) ;;
  *) echo "Refusing unsafe BASE outside qwen36_teacher_ablation: ${BASE}" >&2; exit 2 ;;
esac

if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${NUM_TASKS}" ]]; then
  echo "Array task ${SLURM_ARRAY_TASK_ID} is outside NUM_TASKS=${NUM_TASKS}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Missing model: ${MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -x "${GENERATION_ENV}/bin/python" ]]; then
  echo "Missing generation runtime: ${GENERATION_ENV}/bin/python" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}"

export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=5
export PATH="${GENERATION_ENV}/bin:${PATH}"

COMMON_ARGS=(
  --input-tsv "${INPUT_TSV}"
  --output-root "${OUT_ROOT}"
  --model-path "${MODEL_PATH}"
  --task-id "${SLURM_ARRAY_TASK_ID}"
  --num-tasks "${NUM_TASKS}"
  --tp "${TP}"
)
if [[ -n "${MAX_ROWS}" ]]; then
  COMMON_ARGS+=(--max-rows "${MAX_ROWS}")
fi

echo "method=${METHOD} task=${SLURM_ARRAY_TASK_ID}/${NUM_TASKS} node=$(hostname)"
echo "model=${MODEL_PATH} runtime=${GENERATION_ENV} snapshot=${CODE_SNAPSHOT} started=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

case "${METHOD}" in
  east)
    "${GENERATION_ENV}/bin/python" "${LEGACY_RUNNER}" \
      "${CODE_SNAPSHOT}/llm_output_gigaspeech_trajectory.py" \
      "${COMMON_ARGS[@]}" \
      --batch-size "${BATCH_SIZE:-4096}"
    ;;
  simul-must-c)
    "${GENERATION_ENV}/bin/python" "${LEGACY_RUNNER}" \
      "${CODE_SNAPSHOT}/salami/llm_output_salami.py" \
      "${COMMON_ARGS[@]}" \
      --batch-size "${BATCH_SIZE:-2048}"
    ;;
  *)
    echo "Unknown METHOD=${METHOD}" >&2
    exit 2
    ;;
esac

echo "finished=$(date --iso-8601=seconds)"
