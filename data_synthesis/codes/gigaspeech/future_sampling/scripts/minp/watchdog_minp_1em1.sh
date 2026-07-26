#!/usr/bin/env bash
set -e

# Watchdog: submit the min-p=0.1 (1em1) pipeline as soon as SLURM queue has
# room (MaxJobsPU=10). Keeps enough headroom so other work is not blocked.
#
# Run in background:
#   nohup bash watchdog_minp_1em1.sh > watchdog_1em1.log 2>&1 &

SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/scripts/minp/submit_minp_1em1_pipeline.sh"
MAX_QUEUED=6          # total SLURM jobs allowed before we queue 1em1 (+4 more)
POLL_INTERVAL=300

echo "[watchdog-1em1] start $(date)"

while true; do
  total=$(squeue -u haolingp -h 2>/dev/null | wc -l)
  if (( total <= MAX_QUEUED )); then
    echo "[watchdog-1em1] $(date) — queue has ${total} jobs, submitting 1em1 pipeline"
    bash "${SCRIPT}"
    echo "[watchdog-1em1] done $(date)"
    exit 0
  fi
  echo "[watchdog-1em1] $(date) — queue has ${total} jobs (need ≤${MAX_QUEUED}), sleeping ${POLL_INTERVAL}s"
  sleep "${POLL_INTERVAL}"
done
