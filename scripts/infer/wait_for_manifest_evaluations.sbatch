#!/usr/bin/env bash
#SBATCH --job-name=wait_evaluations
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --requeue
#SBATCH --time=2-00:00:00
#SBATCH -o /home/haolingp/CMU_research_SMT/scripts/infer/slurm_logs/wait_evaluations_%j.out
#SBATCH -e /home/haolingp/CMU_research_SMT/scripts/infer/slurm_logs/wait_evaluations_%j.err

set -euo pipefail

PIPELINE_MANIFEST="${PIPELINE_MANIFEST:?PIPELINE_MANIFEST is required}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
POLL_SECONDS="${POLL_SECONDS:-120}"
KEYS=(comet_eval simultst_eval)

deadline=$((SECONDS + STARTUP_TIMEOUT))
while true; do
  missing=0
  JOB_IDS=()
  for key in "${KEYS[@]}"; do
    value=$(grep -E "^${key}=" "${PIPELINE_MANIFEST}" 2>/dev/null | tail -n 1 | cut -d= -f2- || true)
    if [[ -z "${value}" ]]; then
      missing=$((missing + 1))
    else
      JOB_IDS+=("${value}")
    fi
  done
  if [[ "${missing}" -eq 0 ]]; then
    break
  fi
  if [[ "${SECONDS}" -ge "${deadline}" ]]; then
    echo "Timed out waiting for evaluation IDs in ${PIPELINE_MANIFEST}" >&2
    exit 1
  fi
  sleep 30
done

echo "Waiting for evaluation jobs: ${JOB_IDS[*]}"
while true; do
  active=0
  failed=0
  for job_id in "${JOB_IDS[@]}"; do
    state=$(sacct -X -j "${job_id}" --format=State -n -P | head -n 1 | tr -d '[:space:]')
    state="${state%%+*}"
    case "${state}" in
      COMPLETED)
        ;;
      PENDING|RUNNING|REQUEUED|RESIZING|REVOKED|SUSPENDED|COMPLETING|CONFIGURING|STAGE_OUT)
        active=$((active + 1))
        ;;
      "")
        active=$((active + 1))
        ;;
      *)
        echo "Evaluation job ${job_id} reached terminal state ${state}" >&2
        failed=$((failed + 1))
        ;;
    esac
    echo "$(date --iso-8601=seconds) job=${job_id} state=${state:-UNKNOWN}"
  done
  if [[ "${failed}" -gt 0 ]]; then
    exit 1
  fi
  if [[ "${active}" -eq 0 ]]; then
    echo "All evaluation jobs completed"
    exit 0
  fi
  sleep "${POLL_SECONDS}"
done
