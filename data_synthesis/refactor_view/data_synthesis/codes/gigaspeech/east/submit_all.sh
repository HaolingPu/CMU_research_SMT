#!/usr/bin/env bash
# ============================================================
# EAST — Submit all 4 stages as a chained dependency pipeline.
#
# Usage:
#   bash submit_all.sh          # submit all 4 stages
#   bash submit_all.sh --dry-run  # print sbatch commands without submitting
# ============================================================

set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY=false
[[ "${1}" == "--dry-run" ]] && DRY=true

submit() {
  local desc="$1"; shift
  local cmd=("$@")
  if $DRY; then
    echo "[DRY] ${cmd[*]}" >&2
    echo "9999999"   # fake job ID
  else
    local jid
    # Put --parsable right after sbatch so we get only the job ID (some SLURM
    # treat --parsable after script name as a script arg and then print "Submitted batch job NNN")
    local full=()
    for x in "${cmd[@]}"; do
      full+=("$x")
      [[ "$x" == "sbatch" ]] && full+=("--parsable")
    done
    jid=$("${full[@]}")
    echo "Submitted ${desc}: job ${jid}" >&2
    echo "${jid}"
  fi
}

echo "===== Submitting EAST pipeline ====="

# Stage 1: LLM prep (no dependency)
JID1=$(submit "Stage 1 (LLM prep)" \
  sbatch "${DIR}/stage1_llm.sh")

# Stage 2: Streaming + MetricX input (wait for Stage 1)
JID2=$(submit "Stage 2 (Streaming)" \
  sbatch --dependency=afterok:${JID1} "${DIR}/stage2_streaming.sh")

# Stage 3: MetricX predict, 8-GPU array (wait for Stage 2)
JID3=$(submit "Stage 3 (MetricX array)" \
  sbatch --dependency=afterok:${JID2} --array=0-7 "${DIR}/stage3_metricx.sh")

# Stage 4: Final dataset (wait for ALL array tasks in Stage 3)
JID4=$(submit "Stage 4 (Final)" \
  sbatch --dependency=afterok:${JID3} "${DIR}/stage4_final.sh")

echo ""
echo "  Stage 1 job : ${JID1}"
echo "  Stage 2 job : ${JID2}"
echo "  Stage 3 jobs: ${JID3} (array 0-7)"
echo "  Stage 4 job : ${JID4}"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Cancel all  :  scancel ${JID1} ${JID2} ${JID3} ${JID4}"
