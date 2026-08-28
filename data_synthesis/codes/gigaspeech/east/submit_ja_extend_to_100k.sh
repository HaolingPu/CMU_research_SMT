#!/usr/bin/env bash
# ============================================================
# Extend en→ja EAST to 100k utts (was 40k) using new prompt with
# explicit length-match constraint, then re-run sub-sentence QE chain
# with thr=5.0.
#
# Chain:
#   [1] llm_ja.sh           (8-array; resume mode picks up the new 60k)
#   [2] stage1_ja.sh        (post_process + multi_trajectory, resume mode)
#   [3] subqe orchestrator  (build_consensus_format + segale shards inline,
#                            then sbatches segale-align array → merge → qe prep
#                            → metricx QE 8-array → finalize w/ thr=5.0)
# ============================================================
set -e
HERE=$(cd "$(dirname "$0")" && pwd)

QE_THRESHOLD="${QE_THRESHOLD:-5.0}"

JID_LLM=$(sbatch --parsable "${HERE}/llm_ja.sh")
echo "[1] LLM array (extend to 100k)        : ${JID_LLM}  (array 0-7)"

JID_S1=$(sbatch --parsable --dependency=afterok:${JID_LLM} "${HERE}/stage1_ja.sh")
echo "[2] stage1 (post_process+multi_traj)  : ${JID_S1}   (afterok:${JID_LLM})"

JID_SUBQE=$(sbatch --parsable \
  --dependency=afterok:${JID_S1} \
  --export="ALL,QE_THRESHOLD=${QE_THRESHOLD}" \
  "${HERE}/run_subsentence_qe_orchestrate_ja.sbatch")
echo "[3] sub-QE orchestrator (thr=${QE_THRESHOLD})       : ${JID_SUBQE}  (afterok:${JID_S1})"

cat <<EOF

============================================================
en→ja EAST 100k chain submitted.
  llm_ja extend       : ${JID_LLM}
  stage1              : ${JID_S1}     (afterok:${JID_LLM})
  sub-QE orchestrator : ${JID_SUBQE}  (afterok:${JID_S1})
                        └─ inside this job, segale-align array + downstream
                           jobs get sbatched (segale takes ~12-15h on 8 GPUs)
  thr (sub-QE finalize): ${QE_THRESHOLD}

Watch:
  squeue -u haolingp
  tail -f /data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_ja/slurm_logs/llm_${JID_LLM}_0.out

After completion, final per-latency files:
  outputs/EAST/gigaspeech_ja/segale_qe/final_jsonl_east/<recording>/{low,medium,high}_latency.jsonl

Old 40k results preserved at:
  outputs/EAST/gigaspeech_ja/segale_qe_40k_v1/
============================================================
EOF
