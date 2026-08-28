#!/usr/bin/env bash
set -e

# Watchdog that submits remaining wait-k pipelines when SLURM queue has room.
# MaxJobsPU=10 per user; this script keeps ≤8 wait-k jobs queued so there's
# always headroom for other work.
#
# Run in background:
#   nohup bash watchdog_submit_waitk.sh > watchdog.log 2>&1 &

SCRIPT_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/rule-based-SMT/wait-k/script"
K_TO_SUBMIT=(9 12 15)
MAX_WAITK_QUEUED=6   # leave ~4 slots for other work (bash, QE pipelines from other experiments)
POLL_INTERVAL=300    # 5 minutes

echo "[watchdog] start $(date)"
echo "[watchdog] pending ks: ${K_TO_SUBMIT[*]}"

for K in "${K_TO_SUBMIT[@]}"; do
  while true; do
    waitk_count=$(squeue -u haolingp -h -o "%j" 2>/dev/null | grep -c '^waitk_\|^minp_m' || true)
    if (( waitk_count <= MAX_WAITK_QUEUED )); then
      echo "[watchdog] $(date) — queue has ${waitk_count} waitk+qe jobs, submitting k=${K}"
      bash "${SCRIPT_DIR}/submit_one_pipeline.sh" "${K}"
      sleep 20  # let SLURM register
      break
    fi
    echo "[watchdog] $(date) — queue has ${waitk_count} waitk+qe jobs (need ≤${MAX_WAITK_QUEUED}), sleeping ${POLL_INTERVAL}s"
    sleep "${POLL_INTERVAL}"
  done
done

echo "[watchdog] all pipelines submitted $(date)"
