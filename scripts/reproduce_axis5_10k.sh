#!/usr/bin/env bash
# Submit a 10k old-ASR reproduction of the flagship five-axis consensus pipeline.
# Run this from a Babel login node after pulling the repository.

set -euo pipefail

REPO="/home/haolingp/CMU_research_SMT"
FS="${REPO}/data_synthesis/codes/gigaspeech/future_sampling"
SEG="${FS}/scripts/segale"
RUNS_ROOT="/home/haolingp/slurm_runs"
DATA_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod"

TOTAL_ROWS="${TOTAL_ROWS:-10000}"
NUM_DECODE_TASKS="${NUM_DECODE_TASKS:-4}"
DECODE_CONCURRENCY="${DECODE_CONCURRENCY:-2}"
NUM_POST_SHARDS="${NUM_POST_SHARDS:-24}"
POST_CONCURRENCY="${POST_CONCURRENCY:-24}"
TARGETED_NUM_FUTURES="${TARGETED_NUM_FUTURES:-20}"
MIN_VOTERS_RATIO="${MIN_VOTERS_RATIO:-1.0}"
QE_THRESHOLD="${QE_THRESHOLD:-3.0}"
MIN_RATIO_REF="${MIN_RATIO_REF:-0.7}"
MAX_RATIO_REF="${MAX_RATIO_REF:-1.5}"
SAMPLE_N="${SAMPLE_N:-12500}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

INPUT_TSV="${INPUT_TSV:-/data/group_data/li_lab/haolingp/consensus_handoff/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/reproduce_axis5_10k.sh submit [RUN_TAG]
  bash scripts/reproduce_axis5_10k.sh status RUN_TAG

The submit command queues the complete dependency chain and exits. Defaults can
be overridden with environment variables documented at the top of this script.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_login_node() {
  command -v sbatch >/dev/null || die "sbatch is unavailable; run on a Babel login node"
  [[ "$(hostname)" == login* ]] || die "run on a Babel login node, not $(hostname)"
  [[ -d "${REPO}/.git" ]] || die "missing checkout: ${REPO}"
}

status_run() {
  local run_tag="${1:?RUN_TAG required}"
  local manifest="${RUNS_ROOT}/${run_tag}/run_manifest.txt"
  [[ -f "${manifest}" ]] || die "manifest not found: ${manifest}"
  cat "${manifest}"
  echo
  squeue -u haolingp -o "%.18i %.10P %.26j %.2t %.10M %.10l %R"
}

submit_run() {
  require_login_node

  (( TOTAL_ROWS > 0 )) || die "TOTAL_ROWS must be positive"
  (( NUM_DECODE_TASKS > 0 )) || die "NUM_DECODE_TASKS must be positive"
  (( NUM_POST_SHARDS > 0 )) || die "NUM_POST_SHARDS must be positive"
  (( POST_CONCURRENCY > 0 )) || die "POST_CONCURRENCY must be positive"
  (( POST_CONCURRENCY <= NUM_POST_SHARDS )) || die "POST_CONCURRENCY cannot exceed NUM_POST_SHARDS"
  (( POST_CONCURRENCY <= 24 )) || die "POST_CONCURRENCY cannot exceed the preempt_qos 24-GPU user limit"

  local run_tag="${1:-axis5-oldasr-strict-10k-$(date +%Y%m%d-%H%M%S)}"
  local variant_tag="${run_tag}"
  local exp="gigaspeech-zh-consensus-${variant_tag}-s-bsz4"
  local run_dir="${RUNS_ROOT}/${run_tag}"
  local log_dir="${run_dir}/logs"
  local manifest="${run_dir}/run_manifest.txt"
  local decode_root="${DATA_ROOT}/${run_tag}"
  local post_root="${decode_root}-segale"
  local shards_root="${post_root}/shards"
  local aligned_merged="${post_root}/aligned_all.jsonl"
  local metricx_root="${post_root}/metricx-aligned"
  local qe_filtered="${post_root}/qe${QE_THRESHOLD%.*}-aligned-max"
  local length_filtered="${post_root}/qe${QE_THRESHOLD%.*}-aligned-max-len"
  local ckpts_file="${run_dir}/ckpts.txt"

  [[ ! -e "${run_dir}" ]] || die "run already exists: ${run_dir}"
  mkdir -p "${log_dir}"

  local git_commit
  git_commit=$(git -C "${REPO}" rev-parse HEAD)

  cat > "${manifest}" <<EOF
run_tag=${run_tag}
created=$(date --iso-8601=seconds)
git_commit=${git_commit}
input_tsv=${INPUT_TSV}
total_rows=${TOTAL_ROWS}
num_decode_tasks=${NUM_DECODE_TASKS}
targeted_num_futures=${TARGETED_NUM_FUTURES}
min_voters_ratio=${MIN_VOTERS_RATIO}
future_source_window=1
num_post_shards=${NUM_POST_SHARDS}
post_concurrency=${POST_CONCURRENCY}
qe_threshold=${QE_THRESHOLD}
length_ratio_ref=${MIN_RATIO_REF}:${MAX_RATIO_REF}
sample_n=${SAMPLE_N}
sample_seed=${SAMPLE_SEED}
decode_root=${decode_root}
post_root=${post_root}
length_filtered=${length_filtered}
experiment=${exp}
EOF

  echo "Submitting ${run_tag} from commit ${git_commit}"

  local decode_jid
  decode_jid=$(sbatch --parsable \
    --partition=preempt --qos=preempt_qos --requeue \
    --array="0-$((NUM_DECODE_TASKS - 1))%${DECODE_CONCURRENCY}" \
    --job-name="a5_decode_10k" \
    --output="${log_dir}/decode_%A_%a.out" \
    --error="${log_dir}/decode_%A_%a.err" \
    --export="ALL,INPUT_TSV=${INPUT_TSV},OUTPUT_ROOT=${decode_root},TOTAL_ROWS=${TOTAL_ROWS},NUM_TASKS=${NUM_DECODE_TASKS},TARGETED_NUM_FUTURES=${TARGETED_NUM_FUTURES},MIN_VOTERS_RATIO=${MIN_VOTERS_RATIO},FUTURE_SRC_WINDOW=1" \
    "${FS}/run_J_40k_preempt.sbatch")
  echo "decode=${decode_jid}" | tee -a "${manifest}"

  local prep_jid
  prep_jid=$(sbatch --parsable \
    --dependency="afterok:${decode_jid}" \
    --partition=preempt --qos=preempt_cpu_qos --requeue \
    --job-name="a5_segprep" \
    --output="${log_dir}/segale_prepare_%j.out" \
    --error="${log_dir}/segale_prepare_%j.err" \
    --export="ALL,CONSENSUS_ROOT=${decode_root},OUT_ROOT=${post_root},NUM_DOCS=${TOTAL_ROWS},SYS_ID=${run_tag},NUM_SHARDS=${NUM_POST_SHARDS}" \
    "${SEG}/run_prepare_shards_24.sbatch")
  echo "segale_prepare=${prep_jid}" | tee -a "${manifest}"

  local align_jid
  align_jid=$(sbatch --parsable \
    --dependency="afterok:${prep_jid}" \
    --partition=preempt --qos=preempt_qos --requeue \
    --array="0-$((NUM_POST_SHARDS - 1))%${POST_CONCURRENCY}" \
    --job-name="a5_segale" \
    --output="${log_dir}/segale_%A_%a.out" \
    --error="${log_dir}/segale_%A_%a.err" \
    --export="ALL,SHARDS_ROOT=${shards_root}" \
    "${SEG}/run_segale_align_24gpu_preempt.sbatch")
  echo "segale_align=${align_jid}" | tee -a "${manifest}"

  local merge_jid
  merge_jid=$(sbatch --parsable \
    --dependency="afterok:${align_jid}" \
    --partition=preempt --qos=preempt_cpu_qos --requeue \
    --cpus-per-task=2 --mem=8G --time=00:30:00 \
    --job-name="a5_segmerge" \
    --output="${log_dir}/segale_merge_%j.out" \
    --error="${log_dir}/segale_merge_%j.err" \
    --wrap="source /home/haolingp/miniconda3/etc/profile.d/conda.sh && conda activate /data/user_data/haolingp/conda_envs/segale && PYTHONNOUSERSITE=1 python ${SEG}/merge_aligned_shards.py --shards-root ${shards_root} --output ${aligned_merged} --num-shards ${NUM_POST_SHARDS}")
  echo "segale_merge=${merge_jid}" | tee -a "${manifest}"

  local qe_prep_jid
  qe_prep_jid=$(sbatch --parsable \
    --dependency="afterok:${merge_jid}" \
    --partition=preempt --qos=preempt_cpu_qos --requeue \
    --job-name="a5_qeprep" \
    --output="${log_dir}/qe_prepare_%j.out" \
    --error="${log_dir}/qe_prepare_%j.err" \
    --export="ALL,ALIGNED_FILE=${aligned_merged},CONSENSUS_ROOT=${decode_root},METRICX_RUN_DIR=${metricx_root},NUM_SHARDS=${NUM_POST_SHARDS}" \
    "${SEG}/run_qe_prepare_24.sbatch")
  echo "qe_prepare=${qe_prep_jid}" | tee -a "${manifest}"

  local qe_predict_jid
  qe_predict_jid=$(sbatch --parsable \
    --dependency="afterok:${qe_prep_jid}" \
    --partition=preempt --qos=preempt_qos --requeue \
    --array="0-$((NUM_POST_SHARDS - 1))%${POST_CONCURRENCY}" \
    --job-name="a5_metricx" \
    --output="${log_dir}/metricx_%A_%a.out" \
    --error="${log_dir}/metricx_%A_%a.err" \
    --export="ALL,METRICX_RUN_DIR=${metricx_root}" \
    "${SEG}/run_metricx_qe_24gpu_preempt.sbatch")
  echo "qe_predict=${qe_predict_jid}" | tee -a "${manifest}"

  local qe_finalize_jid
  qe_finalize_jid=$(sbatch --parsable \
    --dependency="afterok:${qe_predict_jid}" \
    --partition=preempt --qos=preempt_cpu_qos --requeue \
    --job-name="a5_qefinal" \
    --output="${log_dir}/qe_finalize_%j.out" \
    --error="${log_dir}/qe_finalize_%j.err" \
    --export="ALL,METRICX_RUN_DIR=${metricx_root},QE_FILTERED_DIR=${qe_filtered},QE_THRESHOLD=${QE_THRESHOLD},NUM_SHARDS=${NUM_POST_SHARDS}" \
    "${SEG}/run_qe_finalize_24.sbatch")
  echo "qe_finalize=${qe_finalize_jid}" | tee -a "${manifest}"

  local length_jid
  length_jid=$(sbatch --parsable \
    --dependency="afterok:${qe_finalize_jid}" \
    --partition=preempt --qos=preempt_cpu_qos --requeue \
    --job-name="a5_length" \
    --output="${log_dir}/length_%j.out" \
    --error="${log_dir}/length_%j.err" \
    --export="ALL,INPUT_DIR=${qe_filtered},OUTPUT_DIR=${length_filtered},MIN_RATIO_REF=${MIN_RATIO_REF},MAX_RATIO_REF=${MAX_RATIO_REF}" \
    "${SEG}/run_length_ratio_filter.sbatch")
  echo "length_filter=${length_jid}" | tee -a "${manifest}"

  local convert_jid
  convert_jid=$(sbatch --parsable \
    --dependency="afterok:${length_jid}" \
    --partition=preempt --qos=preempt_qos --requeue \
    --job-name="a5_convert" \
    --output="${log_dir}/convert_%j.out" \
    --error="${log_dir}/convert_%j.err" \
    --export="ALL,MANIFEST_ROOT=${length_filtered},VARIANT_TAG=${variant_tag},SAMPLE_N=${SAMPLE_N},SAMPLE_SEED=${SAMPLE_SEED}" \
    "${REPO}/scripts/train/run_convert2swift_consensus.sbatch")
  echo "convert=${convert_jid}" | tee -a "${manifest}"

  local train_jid
  train_jid=$(sbatch --parsable \
    --dependency="afterok:${convert_jid}" \
    --partition=preempt --qos=preempt_qos --requeue \
    --job-name="a5_train" \
    --output="${log_dir}/train_%A_%a.out" \
    --error="${log_dir}/train_%A_%a.err" \
    --export="ALL,VARIANT_TAG=${variant_tag}" \
    "${REPO}/scripts/train/train_consensus_s.sh")
  echo "train=${train_jid}" | tee -a "${manifest}"

  local infer_launcher_jid
  infer_launcher_jid=$(sbatch --parsable \
    --dependency="afterok:${train_jid}" \
    --partition=preempt --qos=preempt_qos --requeue \
    --job-name="a5_inferlaunch" \
    --output="${log_dir}/infer_launcher_%j.out" \
    --error="${log_dir}/infer_launcher_%j.err" \
    --export="ALL,EXP=${exp},CHILD_PARTITION=preempt,CHILD_GPU_QOS=preempt_qos,CKPTS_FILE=${ckpts_file}" \
    "${REPO}/scripts/infer/run_infer_after_train_generic.sbatch")
  echo "infer_launcher=${infer_launcher_jid}" | tee -a "${manifest}"

  cat <<EOF

Submitted: ${run_tag}
Manifest : ${manifest}
Outputs  : ${decode_root}
Model    : /data/user_data/haolingp/ckpts/infinisst-omni/${exp}

Status:
  bash ${REPO}/scripts/reproduce_axis5_10k.sh status ${run_tag}
EOF
}

case "${1:-submit}" in
  submit)
    submit_run "${2:-}"
    ;;
  status)
    [[ $# -eq 2 ]] || { usage; exit 2; }
    status_run "$2"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
