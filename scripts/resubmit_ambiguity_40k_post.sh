#!/usr/bin/env bash
# Resubmit the post-decode chain (SEGALE -> MetricX QE -> length filter ->
# convert -> train -> eval launcher) for an existing ambiguity 40k run whose
# decode is complete and verified. Mirrors submit_ambiguity_40k.sh from the
# segale_prepare stage onward, but reads from CONSENSUS_ROOT (normally the
# deduplicated symlink view) and appends to the existing run manifest.

set -euo pipefail

REPO="/home/haolingp/CMU_research_SMT"
FS="${REPO}/data_synthesis/codes/gigaspeech/future_sampling"
SEG="${FS}/scripts/segale"
RUNS_ROOT="/home/haolingp/slurm_runs"
DATA_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod"

RUN_TAG="${1:?usage: resubmit_ambiguity_40k_post.sh RUN_TAG [CONSENSUS_ROOT]}"
run_dir="${RUNS_ROOT}/${RUN_TAG}"
manifest="${run_dir}/run_manifest.txt"
log_dir="${run_dir}/logs"
[[ -f "${manifest}" ]] || { echo "ERROR: manifest not found: ${manifest}" >&2; exit 1; }
[[ "$(hostname)" == login* ]] || { echo "ERROR: run on a BABEL login node" >&2; exit 1; }

decode_root="${DATA_ROOT}/${RUN_TAG}"
consensus_root="${2:-${decode_root}-dedup}"
post_root="${decode_root}-segale"
shards_root="${post_root}/shards"
aligned_merged="${post_root}/aligned_all.jsonl"
metricx_root="${post_root}/metricx-aligned"

TOTAL_ROWS="${TOTAL_ROWS:-40000}"
NUM_POST_SHARDS="${NUM_POST_SHARDS:-24}"
POST_CONCURRENCY="${POST_CONCURRENCY:-24}"
QE_THRESHOLD="${QE_THRESHOLD:-3.0}"
MIN_RATIO_REF="${MIN_RATIO_REF:-0.7}"
MAX_RATIO_REF="${MAX_RATIO_REF:-1.5}"
TRAIN_SAMPLE_N="${TRAIN_SAMPLE_N:-40000}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
qe_filtered="${post_root}/qe${QE_THRESHOLD%.*}-aligned-max"
length_filtered="${post_root}/qe${QE_THRESHOLD%.*}-aligned-max-len"
exp="gigaspeech-zh-consensus-${RUN_TAG}-s-bsz4"
ckpts_file="${run_dir}/ckpts.txt"

(( POST_CONCURRENCY <= 24 )) || { echo "ERROR: post-processing exceeds the 24-GPU limit" >&2; exit 1; }
git_commit=$(git -C "${REPO}" rev-parse HEAD)

cat <<EOF
Run tag        : ${RUN_TAG}
Consensus root : ${consensus_root}
Post root      : ${post_root}
Shards         : ${NUM_POST_SHARDS} (concurrency ${POST_CONCURRENCY})
QE / length    : max sentence QE <= ${QE_THRESHOLD}; ratio ${MIN_RATIO_REF}:${MAX_RATIO_REF}
Train sample   : ${TRAIN_SAMPLE_N} (seed ${SAMPLE_SEED})
Eval           : ACL 6060 dev + Simul-tst-COMMON
Experiment     : ${exp}
Commit         : ${git_commit}
EOF

cat >>"${manifest}" <<EOF
post_resubmit_timestamp=$(date --iso-8601=seconds)
post_resubmit_git_commit=${git_commit}
post_resubmit_consensus_root=${consensus_root}
post_resubmit_reason=decode_complete_40000_verified;stale_chain_afterok_never_satisfiable
EOF

prep_jid=$(sbatch --parsable \
  --partition=preempt --qos=preempt_cpu_qos --requeue \
  --output="${log_dir}/segale_prepare_%j.out" --error="${log_dir}/segale_prepare_%j.err" \
  --export="ALL,CONSENSUS_ROOT=${consensus_root},OUT_ROOT=${post_root},NUM_DOCS=${TOTAL_ROWS},SYS_ID=${RUN_TAG},NUM_SHARDS=${NUM_POST_SHARDS}" \
  "${SEG}/run_prepare_shards_24.sbatch")
echo "post_segale_prepare=${prep_jid}" | tee -a "${manifest}"

align_jid=$(sbatch --parsable --dependency="afterok:${prep_jid}" \
  --partition=preempt --qos=preempt_qos --requeue \
  --array="0-$((NUM_POST_SHARDS - 1))%${POST_CONCURRENCY}" \
  --output="${log_dir}/segale_%A_%a.out" --error="${log_dir}/segale_%A_%a.err" \
  --export="ALL,SHARDS_ROOT=${shards_root}" "${SEG}/run_segale_align_24gpu_preempt.sbatch")
echo "post_segale_align=${align_jid}" | tee -a "${manifest}"

merge_jid=$(sbatch --parsable --dependency="afterok:${align_jid}" \
  --partition=preempt --qos=preempt_cpu_qos --requeue --cpus-per-task=2 --mem=8G --time=00:30:00 \
  --output="${log_dir}/segale_merge_%j.out" --error="${log_dir}/segale_merge_%j.err" \
  --wrap="source /home/haolingp/miniconda3/etc/profile.d/conda.sh && conda activate /data/user_data/haolingp/conda_envs/segale && PYTHONNOUSERSITE=1 python ${SEG}/merge_aligned_shards.py --shards-root ${shards_root} --output ${aligned_merged} --num-shards ${NUM_POST_SHARDS}")
echo "post_segale_merge=${merge_jid}" | tee -a "${manifest}"

qe_prep_jid=$(sbatch --parsable --dependency="afterok:${merge_jid}" \
  --partition=preempt --qos=preempt_cpu_qos --requeue \
  --output="${log_dir}/qe_prepare_%j.out" --error="${log_dir}/qe_prepare_%j.err" \
  --export="ALL,ALIGNED_FILE=${aligned_merged},CONSENSUS_ROOT=${consensus_root},METRICX_RUN_DIR=${metricx_root},NUM_SHARDS=${NUM_POST_SHARDS}" \
  "${SEG}/run_qe_prepare_24.sbatch")
echo "post_qe_prepare=${qe_prep_jid}" | tee -a "${manifest}"

qe_predict_jid=$(sbatch --parsable --dependency="afterok:${qe_prep_jid}" \
  --partition=preempt --qos=preempt_qos --requeue \
  --array="0-$((NUM_POST_SHARDS - 1))%${POST_CONCURRENCY}" \
  --output="${log_dir}/metricx_%A_%a.out" --error="${log_dir}/metricx_%A_%a.err" \
  --export="ALL,METRICX_RUN_DIR=${metricx_root}" "${SEG}/run_metricx_qe_24gpu_preempt.sbatch")
echo "post_qe_predict=${qe_predict_jid}" | tee -a "${manifest}"

qe_finalize_jid=$(sbatch --parsable --dependency="afterok:${qe_predict_jid}" \
  --partition=preempt --qos=preempt_cpu_qos --requeue \
  --output="${log_dir}/qe_finalize_%j.out" --error="${log_dir}/qe_finalize_%j.err" \
  --export="ALL,METRICX_RUN_DIR=${metricx_root},QE_FILTERED_DIR=${qe_filtered},QE_THRESHOLD=${QE_THRESHOLD},NUM_SHARDS=${NUM_POST_SHARDS}" \
  "${SEG}/run_qe_finalize_24.sbatch")
echo "post_qe_finalize=${qe_finalize_jid}" | tee -a "${manifest}"

length_jid=$(sbatch --parsable --dependency="afterok:${qe_finalize_jid}" \
  --partition=preempt --qos=preempt_cpu_qos --requeue \
  --output="${log_dir}/length_%j.out" --error="${log_dir}/length_%j.err" \
  --export="ALL,INPUT_DIR=${qe_filtered},OUTPUT_DIR=${length_filtered},MIN_RATIO_REF=${MIN_RATIO_REF},MAX_RATIO_REF=${MAX_RATIO_REF}" \
  "${SEG}/run_length_ratio_filter.sbatch")
echo "post_length_filter=${length_jid}" | tee -a "${manifest}"

convert_jid=$(sbatch --parsable --dependency="afterok:${length_jid}" \
  --partition=preempt --qos=preempt_cpu_qos --requeue \
  --output="${log_dir}/convert_%j.out" --error="${log_dir}/convert_%j.err" \
  --export="ALL,MANIFEST_ROOT=${length_filtered},VARIANT_TAG=${RUN_TAG},SAMPLE_N=${TRAIN_SAMPLE_N},SAMPLE_SEED=${SAMPLE_SEED}" \
  "${REPO}/scripts/train/run_convert2swift_consensus.sbatch")
echo "post_convert=${convert_jid}" | tee -a "${manifest}"

train_jid=$(sbatch --parsable --dependency="afterok:${convert_jid}" \
  --partition=preempt --qos=preempt_qos --requeue \
  --output="${log_dir}/train_%A_%a.out" --error="${log_dir}/train_%A_%a.err" \
  --export="ALL,VARIANT_TAG=${RUN_TAG}" "${REPO}/scripts/train/train_consensus_s.sh")
echo "post_train=${train_jid}" | tee -a "${manifest}"

eval_launcher_jid=$(sbatch --parsable --dependency="afterok:${train_jid}" \
  --partition=preempt --qos=preempt_cpu_qos --requeue \
  --output="${log_dir}/eval_launcher_%j.out" --error="${log_dir}/eval_launcher_%j.err" \
  --export="ALL,EXP=${exp},CHILD_PARTITION=preempt,CHILD_GPU_QOS=preempt_qos,CKPTS_FILE=${ckpts_file},RUN_SIMULTST=1,CKPTS_SIMULTST_FILE=${run_dir}/ckpts_simultst.txt,PIPELINE_MANIFEST=${manifest}" \
  "${REPO}/scripts/infer/run_infer_after_train_generic.sbatch")
echo "post_eval_launcher=${eval_launcher_jid}" | tee -a "${manifest}"

echo
echo "Resubmitted post-decode chain for ${RUN_TAG}; manifest: ${manifest}"
