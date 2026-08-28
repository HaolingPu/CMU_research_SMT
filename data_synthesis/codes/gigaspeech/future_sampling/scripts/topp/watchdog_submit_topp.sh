#!/usr/bin/env bash
set -e

# Watchdog: submit remaining top-p pipelines when SLURM queue has room.
# MaxJobsPU=10 counts logical submissions (array = 1). Each pipeline = 4.
# Keeps ≤ MAX_QUEUED so other work can still queue.
#
# Run in background:
#   nohup bash watchdog_submit_topp.sh > watchdog_topp.log 2>&1 &

SCRIPT_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/scripts/topp"

# pending (label, value) pairs. Edit if already-submitted set changes.
PENDING=("0p7:0.7" "0p9:0.9" "0p99:0.99")

MAX_QUEUED=6          # logical jobs threshold (leaves 4 slots free for bash + buffer)
POLL_INTERVAL=300

echo "[watchdog-topp] start $(date)"
echo "[watchdog-topp] pending: ${PENDING[*]}"

for pair in "${PENDING[@]}"; do
  label="${pair%%:*}"; val="${pair##*:}"
  while true; do
    total=$(squeue -u haolingp -h -o "%F" 2>/dev/null | sort -u | wc -l)
    if (( total <= MAX_QUEUED )); then
      echo "[watchdog-topp] $(date) — queue has ${total} logical jobs, submitting top-p=${val} (${label})"
      bash "${SCRIPT_DIR}/submit_one_topp_pipeline.sh" "${label}" "${val}"
      sleep 20
      break
    fi
    echo "[watchdog-topp] $(date) — queue has ${total} logical jobs (need ≤${MAX_QUEUED}), sleeping ${POLL_INTERVAL}s"
    sleep "${POLL_INTERVAL}"
  done
done

echo "[watchdog-topp] all pipelines submitted $(date)"
