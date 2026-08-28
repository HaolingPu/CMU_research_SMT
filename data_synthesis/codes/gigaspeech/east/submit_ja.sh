#!/usr/bin/env bash
# Submit the ja pipeline: LLM array (8 workers) -> stage1 (post_process + multi_trajectory).
# stage1 is chained with --dependency=afterok so it only runs if LLM jobs all succeed.

set -e
HERE=$(cd "$(dirname "$0")" && pwd)

JID_LLM=$(sbatch --parsable "${HERE}/llm_ja.sh")
echo "Submitted LLM array       : ${JID_LLM} (array 0-7)"

JID_S1=$(sbatch --parsable --dependency=afterok:${JID_LLM} "${HERE}/stage1_ja.sh")
echo "Submitted stage1 (chained): ${JID_S1}  (waits on ${JID_LLM})"

echo
echo "Watch with:"
echo "  squeue -u haolingp"
echo "  tail -f /data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_ja/slurm_logs/llm_${JID_LLM}_0.out"
