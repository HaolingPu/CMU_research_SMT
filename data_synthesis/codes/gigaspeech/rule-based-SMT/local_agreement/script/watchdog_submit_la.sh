#!/usr/bin/env bash
set -e

# Watchdog: submit remaining LA-N=2 pipelines when the SLURM queue has room.
# MaxJobsPU=10 per user. Keep ≤ MAX_QUEUED jobs in flight so other work can
# still queue up.
#
# Run in background:
#   nohup bash watchdog_submit_la.sh > watchdog_la.log 2>&1 &

SCRIPT_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/rule-based-SMT/local_agreement/script"

# List of segment sizes still to submit. Edit if already-submitted set changes.
SEGS_TO_SUBMIT=(3 4 5)
MAX_QUEUED=6          # keep ≤6 LA/QE jobs in flight (leaves 4 slots free)
POLL_INTERVAL=300     # 5 min

echo "[watchdog-la] start $(date)"
echo "[watchdog-la] pending segs: ${SEGS_TO_SUBMIT[*]}"

for SEG in "${SEGS_TO_SUBMIT[@]}"; do
  while true; do
    # Count LA gen jobs + reused minp_m* QE jobs (prepare/predict/finalize)
    count=$(squeue -u haolingp -h -o "%j" 2>/dev/null | grep -c '^la2_seg\|^minp_m' || true)
    if (( count <= MAX_QUEUED )); then
      echo "[watchdog-la] $(date) — queue has ${count} la+qe jobs, submitting seg=${SEG}"
      bash "${SCRIPT_DIR}/submit_one_la_pipeline.sh" "${SEG}"
      sleep 20
      break
    fi
    echo "[watchdog-la] $(date) — queue has ${count} la+qe jobs (need ≤${MAX_QUEUED}), sleeping ${POLL_INTERVAL}s"
    sleep "${POLL_INTERVAL}"
  done
done

echo "[watchdog-la] all pipelines submitted $(date)"
