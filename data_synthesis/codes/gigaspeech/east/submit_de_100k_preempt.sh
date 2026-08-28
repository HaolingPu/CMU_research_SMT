#!/usr/bin/env bash
# ============================================================
# Full en->de EAST 100k chain (preempt, 24 GPUs):
#   [1] llm_de.sh           (24-array preempt, 100k utts)
#   [2] stage1_de.sh        (preempt: post_process + multi_trajectory)
#   [3] subqe orchestrator  (preempt: build_consensus_format + segale shards inline,
#                            then sbatches segale-align 24x -> merge -> qe prep
#                            -> metricx QE 8x -> finalize w/ thr=QE_THRESHOLD)
# ============================================================
set -e
HERE=$(cd "$(dirname "$0")" && pwd)

QE_THRESHOLD="${QE_THRESHOLD:-3.0}"

JID_LLM=$(sbatch --parsable "${HERE}/llm_de.sh")
echo "[1] LLM array (24-task preempt, 100k) : ${JID_LLM}  (array 0-23)"

JID_S1=$(sbatch --parsable --dependency=afterok:${JID_LLM} "${HERE}/stage1_de.sh")
echo "[2] stage1 (post_process+multi_traj)   : ${JID_S1}   (afterok:${JID_LLM})"

JID_SUBQE=$(sbatch --parsable \
  --dependency=afterok:${JID_S1} \
  --export="ALL,QE_THRESHOLD=${QE_THRESHOLD}" \
  "${HERE}/run_subsentence_qe_orchestrate_de.sbatch")
echo "[3] sub-QE orchestrator (thr=${QE_THRESHOLD})       : ${JID_SUBQE}  (afterok:${JID_S1})"

cat <<EOF

============================================================
en->de EAST 100k chain submitted (preempt).
  llm_de (24-array)   : ${JID_LLM}
  stage1              : ${JID_S1}     (afterok:${JID_LLM})
  sub-QE orchestrator : ${JID_SUBQE}  (afterok:${JID_S1})
                        +- inside: segale-align 24x -> merge -> qe_prep
                                   -> metricx QE 8x -> finalize
  thr (sub-QE finalize): ${QE_THRESHOLD}

Watch:
  squeue -u haolingp
  tail -f /data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_de/slurm_logs/llm_${JID_LLM}_0.out

After completion, final per-latency files:
  outputs/EAST/gigaspeech_de/segale_qe/final_jsonl_east/<recording>/{low,medium,high}_latency.jsonl
============================================================
EOF
